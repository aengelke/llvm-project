//===- EhFrame.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_EHFRAME_H
#define LLD_ELF_EHFRAME_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace lld::elf {
struct EhSectionPiece;

uint8_t getFdeEncoding(EhSectionPiece *p);
bool hasLSDA(const EhSectionPiece &p);
bool isCompactUnwind(const EhSectionPiece &p);

struct CompactUnwindDescriptor {
  uint64_t off;
  uint64_t desc;
};
// Returns address range.
uint64_t readCompactUnwindDescriptors(
    const EhSectionPiece &fde, uint8_t fdeEnc, bool isCompactUnwind,
    llvm::SmallVectorImpl<CompactUnwindDescriptor> &descs);
}

#endif
