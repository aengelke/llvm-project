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

namespace lld::elf {
struct Ctx;
struct EhSectionPiece;
struct CompactUnwindDescriptor {
  uint64_t off;
  uint64_t desc;
};

/// Parse CIE and init the cie union member of the EhSectionPiece.
void parseCIE(EhSectionPiece &cie);

/// Try to encode an FDE using compact unwind descriptors. If encoding failed,
/// the compactUnwindDescriptors array is empty. In any case, this extracts the
/// address range from the FDE.
void encodeAsCompactUnwind(Ctx &ctx, const EhSectionPiece &cie,
                           EhSectionPiece &fde);
}

#endif
