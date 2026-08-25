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

  uint8_t readByte();
  void skipBytes(size_t count);
  StringRef readString();
  uint64_t readULeb128();
  int64_t readSLeb128();
  void skipAugP();

  InputSectionBase *isec;
  ArrayRef<uint8_t> d;
};
}

// Read a byte and advance D by one byte.
uint8_t EhReader::readByte() {
  if (d.empty()) {
    errOn(d.data(), "unexpected end of CIE");
    return 0;
  }
  uint8_t b = d.front();
  d = d.slice(1);
  return b;
}

void EhReader::skipBytes(size_t count) {
  if (d.size() < count)
    errOn(d.data(), "CIE is too small");
  else
    d = d.slice(count);
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

  // TODO: actually try encoding.
  fde.u.fde.cuDescs = nullptr;
  fde.u.fde.cuDescSize = 0;
}
