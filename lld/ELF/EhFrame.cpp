//===- EhFrame.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// .eh_frame section contains information on how to unwind the stack when
// an exception is thrown. The section consists of sequence of CIE and FDE
// records. The linker needs to merge CIEs and associate FDEs to CIEs.
// That means the linker has to understand the format of the section.
//
// This file contains a few utility functions to read .eh_frame contents.
//
//===----------------------------------------------------------------------===//

#include "EhFrame.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Target.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/LEB128.h"

#define DEBUG_TYPE "lld-ehframe"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::dwarf;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

namespace {
struct EhReader {
  EhReader(InputSectionBase *s, ArrayRef<uint8_t> d) : isec(s), d(d) {}

  template <class P> void errOn(const P *loc, const Twine &msg) {
    Ctx &ctx = isec->file->ctx;
    Err(ctx) << "corrupted .eh_frame: " << msg << "\n>>> defined in "
             << isec->getObjMsg((const uint8_t *)loc - isec->content().data());
  }

  bool empty() const { return d.empty(); }

  const uint8_t *takeN(size_t count) {
    if (d.size() < count)
      errOn(d.data(), "CIE/FDE is too small");
    const uint8_t *res = d.data();
    d = d.slice(count);
    return res;
  }

  uint8_t readByte() { return *takeN(1); }
  uint16_t read16() { return ::read16(isec->file->ctx, takeN(2)); }
  uint32_t read32() { return ::read32(isec->file->ctx, takeN(4)); }
  void skipBytes(size_t count) { takeN(count); }
  StringRef readString();
  uint64_t readULeb128();
  int64_t readSLeb128();
  void skipAugP();

  InputSectionBase *isec;
  ArrayRef<uint8_t> d;
};
}

// Read a null-terminated string.
StringRef EhReader::readString() {
  const uint8_t *end = llvm::find(d, '\0');
  if (end == d.end()) {
    errOn(d.data(), "corrupted CIE (failed to read string)");
    return {};
  }
  StringRef s = toStringRef(d.slice(0, end - d.begin()));
  d = d.slice(s.size() + 1);
  return s;
}

uint64_t EhReader::readULeb128() {
  const char *err = nullptr;
  const uint8_t *p = d.data();
  uint64_t ret = decodeULEB128AndInc(p, d.end(), &err);
  if (err)
    errOn(p, "corrupted .eh_frame (failed to read LEB128)");
  else
    d = d.slice(p - d.data());
  return ret;
}

int64_t EhReader::readSLeb128() {
  const char *err = nullptr;
  const uint8_t *p = d.data();
  int64_t ret = decodeSLEB128AndInc(p, d.end(), &err);
  if (err)
    errOn(p, "corrupted .eh_frame (failed to read LEB128)");
  else
    d = d.slice(p - d.data());
  return ret;
}

static size_t getEncodingSize(Ctx &ctx, unsigned enc) {
  switch (enc & 0x0f) {
  case DW_EH_PE_absptr:
  case DW_EH_PE_signed:
    return ctx.arg.wordsize;
  case DW_EH_PE_udata2:
  case DW_EH_PE_sdata2:
    return 2;
  case DW_EH_PE_udata4:
  case DW_EH_PE_sdata4:
    return 4;
  case DW_EH_PE_udata8:
  case DW_EH_PE_sdata8:
    return 8;
  }
  return 0;
}

void EhReader::skipAugP() {
  uint8_t enc = readByte();
  if ((enc & 0xf0) == DW_EH_PE_aligned)
    return errOn(d.data() - 1, "DW_EH_PE_aligned encoding is not supported");
  size_t size = getEncodingSize(isec->getCtx(), enc);
  if (size == 0)
    return errOn(d.data() - 1, "unknown FDE encoding");
  if (size >= d.size())
    return errOn(d.data() - 1, "corrupted CIE");
  d = d.slice(size);
}

void elf::parseCIE(EhSectionPiece &p) {
  EhReader reader(p.sec, p.data());
  reader.skipBytes(8);
  int version = reader.readByte();
  if (version != 1 && version != 3) {
    reader.errOn(reader.d.data() - 1,
                 "FDE version 1 or 3 expected, but got " + Twine(version));
    return;
  }

  StringRef aug = reader.readString();

  p.u.cie.fdeEncoding = DW_EH_PE_absptr;
  p.u.cie.hasPersonality = false;
  p.u.cie.hasLSDA = false;
  p.u.cie.codeAlignmentFactor = reader.readULeb128();
  p.u.cie.dataAlignmentFactor = reader.readSLeb128();

  // Skip the return address register. In CIE version 1 this is a single
  // byte. In CIE version 3 this is an unsigned LEB128.
  if (version == 1)
    reader.readByte();
  else
    reader.readULeb128();

  for (char c : aug) {
    switch (c) {
    case 'z':
      reader.readULeb128();
      break;
    case 'P':
      p.u.cie.hasPersonality = true;
      reader.skipAugP();
      break;
    case 'L':
      p.u.cie.hasLSDA = true;
      reader.readByte();
      break;
    case 'R':
      p.u.cie.fdeEncoding = reader.readByte();
      break;
    case 'B':
    case 'S':
    case 'G':
      break;
    default:
      reader.errOn(aug.data(), "unknown .eh_frame augmentation string: " + aug);
    }
  }

  p.u.cie.cfiBegin = p.data().size() - reader.d.size();
}

struct CFIState {
  using Reg = uint8_t;

  uint64_t cfaOff = 0;   ///< CFA offset (unscaled).
  Reg cfaReg = Reg(-1u); ///< CFA base register.
  /// Saved register at offset (scaled by data alignment factor).
  SmallVector<std::pair<Reg, uint64_t>, 8> regs;

  void defCFARegister(Reg reg) {
    cfaReg = reg;
  }

  void defCFAOffset(uint64_t offset) {
    cfaOff = offset;
  }

  void store(Reg reg, uint64_t scaledOff) {
    for (auto &entry : regs) {
      if (entry.first == reg) {
        entry.second = scaledOff;
        return;
      }
    }
    regs.emplace_back(reg, scaledOff);
  }

  void restore(Reg reg) {
    for (unsigned i = 0; i != regs.size(); i++) {
      if (regs[i].first == reg) {
        std::swap(regs[i], regs.back());
        regs.pop_back();
        return;
      }
    }
  }

  void print(raw_ostream &os) const {
    os << "CFIState{r" << unsigned(cfaReg) << "+" << cfaOff;
    for (const auto &[reg, off] : regs)
      os << ",r" << unsigned(reg) << "@" << off;
    os << "}";
  }
};

class CUBuilderX86 {
  SmallVector<CompactUnwindDescriptor, 4> descs;
  const EhSectionPiece &fde;
  const EhSectionPiece &cie;

  size_t curOffset = 0;
  CFIState curState;
  SmallVector<CFIState, 0> stateStack;

public:
  CUBuilderX86(const EhSectionPiece &fde, const EhSectionPiece &cie)
      : fde(fde), cie(cie) {}

  ArrayRef<CompactUnwindDescriptor> getDescs() const { return descs; }

  static std::optional<uint64_t> encode(const CFIState &curState) {
    enum : uint8_t {
      RAX = 0,
      RBX = 3,
      RBP = 6,
      RSP = 7,
      R12 = 12,
      R13 = 13,
      R14 = 14,
      R15 = 15,
      RIP = 16,
    };

    if (curState.cfaOff % 8 != 0)
      return std::nullopt;

    CFIState::Reg regs[7] = {};
    for (const auto &[reg, off] : curState.regs) {
      if (off == 0 || off > 8)
        return std::nullopt;
      if (curState.cfaReg == RSP && off >= curState.cfaOff)
        continue;
      regs[off - 1] = reg;
    }
    if (regs[0] != RIP)
      return std::nullopt;
    // RBP can either be at the top of the stack frame (LLVM) or between R12 and
    // RBX (GCC). Support both.
    static constexpr CFIState::Reg saveOrder[] = {RBP, R15, R14, R13, R12, RBP, RBX};
    unsigned regMask = 0;
    unsigned regCount = 1; // RIP
    for (unsigned i = 0; i < 7; i++) {
      if (regs[regCount] == saveOrder[i]) {
        regMask |= 1 << i;
        regCount += 1;
      }
    }
    if (regCount != curState.regs.size())
      return std::nullopt;

    if (curState.cfaReg == RBP) {
      if (curState.cfaOff != 16)
        return std::nullopt;
      // mode:3, personality:3, prologue_size:8, reserved:12, saved_regs:6.
      return (1u << 29) | regMask;
    }

    if (curState.cfaReg != RSP)
      return std::nullopt;

    // We ignore all callee-saved registers below RSP.
    if (curState.cfaOff == 8)
      return 0; // Empty frame.

    // mode:3, personality:3, frame_size:19, saved_regs:7.
    // Lowest 3 bits of cfaOff are known to be zero, checked above.
    if ((curState.cfaOff >> 3) >= uint64_t{1} << 19)
      return std::nullopt;
    return (2u << 29) | (curState.cfaOff << (7 - 3)) | regMask;
  }

  bool handleAdvance(size_t delta) {
    delta *= cie.u.cie.codeAlignmentFactor;
    LLVM_DEBUG(dbgs() << "advance " << curOffset << "+" << delta << " ";
               curState.print(dbgs());
               dbgs() << "\n";);
    std::optional<uint64_t> desc = encode(curState);
    if (!desc) {
      LLVM_DEBUG(dbgs() << "FAIL: unable to encode\n");
      return true;
    }
    descs.push_back(CompactUnwindDescriptor{curOffset, *desc});
    curOffset += delta;
    if (curOffset > fde.u.fde.addrRange) {
      LLVM_DEBUG(dbgs() << "FAIL: out of address range?\n");
      return true;
    }
    return false;
  }

  bool addCFIInstrs(EhReader &reader, bool isCIE) {
    while (!reader.empty()) {
      unsigned opcode = reader.readByte();
      constexpr uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
      constexpr uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;
      switch (opcode & DWARF_CFI_PRIMARY_OPCODE_MASK) {
      case DW_CFA_advance_loc:
        if (handleAdvance(opcode & DWARF_CFI_PRIMARY_OPERAND_MASK))
          return true;
        break;
      case DW_CFA_offset:
        curState.store(opcode & DWARF_CFI_PRIMARY_OPERAND_MASK, reader.readULeb128());
        break;
      case DW_CFA_restore:
        curState.restore(opcode & DWARF_CFI_PRIMARY_OPERAND_MASK);
        break;
      default:
        switch (opcode) {
        case DW_CFA_advance_loc1:
          if (handleAdvance(reader.readByte()))
            return true;
          break;
        case DW_CFA_advance_loc2:
          if (handleAdvance(reader.read16()))
            return true;
          break;
        case DW_CFA_advance_loc4:
          if (handleAdvance(reader.read32()))
            return true;
          break;
        case DW_CFA_remember_state:
          stateStack.push_back(curState);
          break;
        case DW_CFA_restore_state:
          if (stateStack.empty())
            return true;
          curState = stateStack.pop_back_val();
          break;
        case DW_CFA_def_cfa:
          curState.defCFARegister(reader.readULeb128());
          curState.defCFAOffset(reader.readULeb128());
          break;
        case DW_CFA_def_cfa_register:
          curState.defCFARegister(reader.readULeb128());
          break;
        case DW_CFA_def_cfa_offset:
          curState.defCFAOffset(reader.readULeb128());
          break;
        case DW_CFA_GNU_args_size:
          if (uint64_t argSize = reader.readULeb128()) {
            LLVM_DEBUG(dbgs() << "FAIL: GNU_args_size " << argSize << "\n");
            return true;
          }
          break;
        case DW_CFA_nop:
          break;
        default:
          LLVM_DEBUG(dbgs() << "FAIL: unhandled opcode " << opcode << "\n");
          return true;
        }
      }
    }
    return false;
  }

  bool finalize() {
    // Advance to keep state of most recent CFI instructions.
    if (handleAdvance(fde.u.fde.addrRange - curOffset))
      return true;
    // So that we can easily compute the range of a descriptor.
    descs.push_back(CompactUnwindDescriptor{fde.u.fde.addrRange, 1});
    unsigned n = 0;
    for (unsigned i = 0; i < descs.size() - 1; i++) {
      descs[n++] = descs[i];
      uint64_t next = descs[i + 1].desc;
      size_t size = descs[i + 1].off - descs[n - 1].off;
      uint64_t &desc = descs[n - 1].desc;

      // We generate all descriptors without prologue/epilogue compression.
      assert(desc >> 32 == 0 && "desc should not have prologue/epilogue yet");
      if (desc == 0) {
        if (size < 256 && ((next >> 29) & 7) == 2) {
          // We can fold a NULL descriptor into the RSP prologue if there's no
          // CSR (so could be sub rsp,xxx) or exactly one CSR with frame size 8.
          unsigned csrs = next & 0x7f;
          unsigned cfaSize = (next >> 7) & 0x7ffff;
          if (csrs == 0 || ((csrs & (csrs - 1)) == 0 && cfaSize == 2)) {
            desc = next | size << 32;
            i += 1;
            // Immediately fetch next descriptor, if any, to continue folding
            // RSP-based prologue sequences.
            if (i < descs.size() - 1) {
              next = descs[i + 1].desc;
              size = descs[i + 1].off - descs[n - 1].off;
            } else {
              continue;
            }
          }
        }
      }
      // RSP-based frame.
      if (((desc >> 29) & 7) == 2) {
        // We either have a short prologue from a folded NULL descriptor or no
        // prologue at all. We have no epilogue.
        //
        // First, try to build a prologue chain. This is non-trivial, as we
        // don't know which CSRs the frame has, all we see is:
        //       CFA=rsp+16                        <<<< desc
        //    +2 CFA=rsp+24
        //    +1 CFA=rsp+32 R15=[CFA-16] R14=[CFA-24]
        // Or this:
        //       CFA=rsp+16                        <<<< desc
        //    +2 CFA=rsp+24
        //    +1 CFA=rsp+32 R15=[CFA-16] R14=[CFA-24] RBX=[CFA-32]
        // Or this (GCC):
        //       CFA=rsp+16                        <<<< desc
        //    +6 CFA=rsp+24
        //    +1 CFA=rsp+32 R15=[CFA-16] R14=[CFA-24] RBX=[CFA-32]
        // Or also this (if a previous frame didn't merge us, e.g. with GCC's
        // separate shrink wrapping):
        //       CFA=rsp+32                        <<<< desc
        //    +2 CFA=rsp+40
        //
        // We first look forward to find growing stack frames to determine the
        // set of CSRs. We stop when the advance is unreasonably large (>=8) or
        // after the stack frame growth is not 8 (which must be some sub rsp,x).
        unsigned csrs = desc & 0x7f;
        unsigned frameSize = (desc >> 7) & 0x7ffff;
        unsigned lastFrameSizeDelta = 0;
        unsigned j = 0;
        uint8_t advances[8];
        unsigned totalAdvances = 0;
        for (; i + j + 1 < descs.size() && j < 8; j++) {
          uint64_t advance = descs[i + 1 + j].off - descs[i + j].off;
          if (advance >= 8)
            break;
          uint64_t jdesc = descs[i + 1 + j].desc;
          // If this is not an RSP frame, it clears some previously saved regs
          // or the frame size doesn't grow, stop.
          if (((jdesc >> 29) & 7) != 2 || (jdesc | csrs) != jdesc)
            break;
          unsigned newFrameSize = (jdesc >> 7) & 0x7ffff;
          if (newFrameSize <= frameSize)
            break;
          unsigned frameSizeDelta = newFrameSize - frameSize;
          if (frameSizeDelta == 1 && advance > 2)
            break; // PUSH is 1 or 2 bytes.
          advances[j] = advance;
          totalAdvances += advance;
          frameSize = newFrameSize;
          lastFrameSizeDelta = frameSizeDelta;
          csrs = jdesc & 0x7f;
          // If we grow by more than a push, stop.
          if (frameSizeDelta != 1) {
            j++;
            break;
          }
        }
        if (j > 0) {
          LLVM_DEBUG(dbgs() << "FOLD RSP: " << format("%016x j=%u fs=%u csr=%02x", desc, j, frameSize, csrs); for (unsigned k = 0; k < j; k++) dbgs() << format(" +%d:%08x", advances[k], descs[i+k+1].desc); dbgs() << "\n";);
          // Now we must verify that the advances match our expectations.
          unsigned numCsrs = popcount(csrs);
          unsigned subRspDelta = frameSize - numCsrs - 1;
          unsigned k = j;
          if (subRspDelta) {
            if (subRspDelta != lastFrameSizeDelta) {
              LLVM_DEBUG(dbgs() << "FOLD RSP SKIP: last frame size delta\n");
              goto skipPrologue;
            }
            // Expected instruction size.
            unsigned subRspSize = subRspDelta == 1 ? 1 : subRspDelta < 0x10 ? 4 : 7;
            if (advances[k - 1] != subRspSize) {
              LLVM_DEBUG(dbgs() << "FOLD RSP SKIP: sub rsp size\n");
              goto skipPrologue;
            }
            k -= 1;
          }
          // The last k CSRs advance sizes must match corresponding push
          // instructions. The first push/CSR is never relevant.
          if (k + 1 > numCsrs) {
            LLVM_DEBUG(dbgs() << "FOLD RSP SKIP: num k\n");
            goto skipPrologue;
          }
          // TODO: optimize?
          uint8_t regs[7] = {0};
          for (unsigned l = 0, m = 0; l < 7; l++)
            if (csrs & (1 << l))
              regs[m++] = l;
          for (unsigned l = 0; l < k; l++) {
            unsigned reg = regs[numCsrs - k + l];
            unsigned pushSize = (reg >= 1 && reg <= 4) ? 2 : 1;
            if (advances[l] != pushSize) {
              LLVM_DEBUG(dbgs() << format("FOLD RSP SKIP: push size l=%u k=%u reg=%u pushSize=%u\n", l, k, reg, pushSize));
              goto skipPrologue;
            }
          }
          LLVM_DEBUG(dbgs() << "FOLD RSP SUCCESS\n");
          uint64_t prologueSize = ((desc >> 32) & 0xff) + totalAdvances;
          if (prologueSize < 256) {
            i += j;
            desc = descs[i].desc | (prologueSize << 32);
            // Immediately fetch next descriptor, if any, to continue folding
            // RSP-based epilogue sequences.
            if (i < descs.size() - 1) {
              next = descs[i + 1].desc;
              size = descs[i + 1].off - descs[n - 1].off;
            } else {
              continue;
            }
          }
        }
skipPrologue:;
        // Now, we try folding an epilogue sequence. This is easier, as we know
        // all CSRs now.
        // First, make sure that saved values are correct in case we aborted
        // prologue folding.
        csrs = desc & 0x7f;
        frameSize = (desc >> 7) & 0x7ffff;
        unsigned numCsrs = popcount(csrs);
        // If there's a sub rsp, it doesn't matter, because it comes first.
        // Otherwise, restoring the first CSR doesn't matter.
        if (frameSize <= numCsrs) {
          // TODO: handle. Can probably happen if there are stale CSRs below
          // RSP due to DWARF "optimization".
          LLVM_DEBUG(dbgs() << "FOLD RSP EPILOGUE too many CSRs\n");
          continue;
        }
        bool hasSubRsp = frameSize > numCsrs + 1;
        // TODO: optimize?
        uint8_t regs[7] = {0};
        for (unsigned l = 0, m = 0; l < 7; l++)
          if (csrs & (1 << l))
            regs[m++] = l;
        // We simply try to fold subsequent descriptors into the current
        // descriptor. The current descriptor has no epilogue yet. The
        // requirements are:
        //  - frame size of the next descriptor must just hold the CSRs.
        //  - the address range(!) of that descriptor must match the pop size
        //    of the CSR. The advance from the current descriptor is irrelevant.
        //    (The range must match, because we can't just arbitrarily stop and
        //    hold the current state in epilogue sequences. For a sequence like
        //    add rsp, X; <DESC> pop r12; <DESC> nop; nop; pop r13 <DESC>, we
        //    need a new descriptor after the pop r12.)
        unsigned epilogueSize = 0;
        unsigned count = 0;
        while (i < descs.size() - 2) {
          uint64_t jdesc = descs[i + 1].desc;
          LLVM_DEBUG(dbgs() << format("FOLD RSP EP: desc=%08x idesc=%08x jdesc=%08x\n", desc, descs[i].desc, jdesc));
          // If this is not an RSP frame, stop.
          if (((jdesc >> 29) & 7) != 2 && jdesc != 0)
            break;
          unsigned newFrameSize = jdesc ? (jdesc >> 7) & 0x7ffff : 1;
          if (newFrameSize != numCsrs - count + hasSubRsp) {
            LLVM_DEBUG(dbgs() << format("FOLD RSP SKIP EP: newFrameSize=%u numCsrs=%u count=%u\n", newFrameSize, numCsrs, count));
            break;
          }
          uint64_t range = descs[i + 2].off - descs[i + 1].off;
          if (newFrameSize > 1) {
            unsigned reg = regs[newFrameSize - 2];
            unsigned pushSize = (reg >= 1 && reg <= 4) ? 2 : 1;
            if (range != pushSize) {
              LLVM_DEBUG(dbgs() << format("FOLD RSP SKIP EP: push size range=%u reg=%u pushSize=%u newFrameSize=%u numCsrs=%u\n", range, reg, pushSize, newFrameSize, numCsrs));
              break;
            }
          }
          if (epilogueSize + range >= 256)
            break;
          epilogueSize += range;
          count += 1;
          i += 1;
        }
        desc |= (uint64_t)epilogueSize << 40;
        uint64_t size = descs[i + 1].off - descs[n - 1].off;
        if (1 || (size && (desc >> 40 & 0xff) + (desc >> 32 & 0xff) >= size)) {
          LLVM_DEBUG(dbgs() << format("FOO? %016lx size=%zx\n", desc, size));
        }
      }
    }
    descs.truncate(n);
    for (size_t i = 0; i < descs.size() - 1; i++)
      assert(descs[i].off < descs[i + 1].off);
    return false;
  }
};

void elf::encodeAsCompactUnwind(Ctx &ctx, const EhSectionPiece &cie, EhSectionPiece &fde) {
  EhReader reader(fde.sec, fde.data());
  size_t encSize = getEncodingSize(ctx, cie.u.cie.fdeEncoding);
  // Skip length, cie_pointer, initial_location.
  reader.skipBytes(8 + encSize);
  // Read address_range.
  if (encSize == 4)
    fde.u.fde.addrRange = read32(ctx, reader.d.data());
  else if (encSize == 8)
    fde.u.fde.addrRange = read64(ctx, reader.d.data());
  else
    assert(0 && "wait, what?");
  reader.skipBytes(encSize);
  // Skip augmentation length and data.
  // TODO: only when z is present in CIE augmentation.
  uint64_t augLen = reader.readULeb128();
  reader.skipBytes(augLen);

  // Default-initialize to non-compact encoding in case of failures.
  fde.u.fde.cuDescs = nullptr;
  fde.u.fde.cuDescSize = 0;

  CUBuilderX86 cub(fde, cie);
  EhReader cieReader(cie.sec, cie.data().slice(cie.u.cie.cfiBegin));
  if (cub.addCFIInstrs(cieReader, /*isCIE=*/true))
    return;
  if (cub.addCFIInstrs(reader, /*isCIE=*/false))
    return;
  if (cub.finalize())
    return;

  size_t allocSz = sizeof(CompactUnwindDescriptor) * cub.getDescs().size();
  void *alloc = ctx.bAlloc.Allocate(allocSz, alignof(CompactUnwindDescriptor));
  memcpy(alloc, cub.getDescs().data(), allocSz);
  fde.u.fde.cuDescs = reinterpret_cast<CompactUnwindDescriptor *>(alloc);
  fde.u.fde.cuDescSize = cub.getDescs().size();
}
