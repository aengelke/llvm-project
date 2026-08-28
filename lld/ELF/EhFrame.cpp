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
