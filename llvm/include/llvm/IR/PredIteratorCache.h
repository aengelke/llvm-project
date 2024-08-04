//===- PredIteratorCache.h - pred_iterator Cache ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PredIteratorCache class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PREDITERATORCACHE_H
#define LLVM_IR_PREDITERATORCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

/// PredIteratorCache - This class is an extremely trivial cache for
/// predecessor iterator queries.  This is useful for code that repeatedly
/// wants the predecessor list for the same blocks.
class PredIteratorCache {
  /// Storage, indexed by block number.
  SmallVector<ArrayRef<BasicBlock *>> Storage;
  /// Whether storage is valid, indexed by block number.
  BitVector StorageValid;
  /// Block number epoch to guard against renumberings.
  unsigned BlockNumberEpoch;

  /// Memory - This is the space that holds cached preds.
  BumpPtrAllocator Memory;

public:
  size_t size(BasicBlock *BB) { return get(BB).size(); }
  ArrayRef<BasicBlock *> get(BasicBlock *BB) {
#ifndef NDEBUG
    // In debug builds, verify that no renumbering has occured.
    if (StorageValid.empty())
      BlockNumberEpoch = BB->getParent()->getBlockNumberEpoch();
    else
      assert(BlockNumberEpoch == BB->getParent()->getBlockNumberEpoch() &&
             "Blocks renumbered during lifetime of PredIteratorCache");
#endif

    if (LLVM_LIKELY(BB->getNumber() < StorageValid.size()))
      if (StorageValid[BB->getNumber()])
        return Storage[BB->getNumber()];

    if (BB->getNumber() >= StorageValid.size()) {
      StorageValid.resize(BB->getParent()->getMaxBlockNumber());
      // Don't initialize memory for performance
      Storage.resize_for_overwrite(BB->getParent()->getMaxBlockNumber());
    }

    SmallVector<BasicBlock *, 32> PredCache(predecessors(BB));
    BasicBlock **Data = Memory.Allocate<BasicBlock *>(PredCache.size());
    std::copy(PredCache.begin(), PredCache.end(), Data);
    StorageValid[BB->getNumber()] = true;
    return Storage[BB->getNumber()] = ArrayRef(Data, PredCache.size());
  }

  /// clear - Remove all information.
  void clear() {
    StorageValid.clear();
    Memory.Reset();
  }
};

} // end namespace llvm

#endif
