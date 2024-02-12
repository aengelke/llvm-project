//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRUCTIONINLINESTORAGE_H
#define LLVM_IR_INSTRUCTIONINLINESTORAGE_H

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// Base class for exposing Instruction::AuxData. Don't use directly.
class InstructionInlineStorage {
protected:
  const Function *F;

  InstructionInlineStorage(const Function *Func) : F(Func) {
    if (!F)
      return;
    if (F->InstAuxDataUsed)
      reportFatalInternalError("conflicting InstructionInlineStorage use");
    F->InstAuxDataUsed = true;
  }
  ~InstructionInlineStorage() {
    assert(!F && "must call release() before destructing");
  }

  InstructionInlineStorage(const InstructionInlineStorage &) = delete;
  InstructionInlineStorage(InstructionInlineStorage &&Other) : F(Other.F) {
    Other.F = nullptr;
  }
  InstructionInlineStorage &
  operator=(const InstructionInlineStorage &) = delete;
  InstructionInlineStorage &operator=(InstructionInlineStorage &&Other) {
    if (F)
      release();
    F = Other.F;
    Other.F = nullptr;
    return *this;
  }

  void clearToZero() {
    if (F->InstAuxDataMax == 0)
      return;
    for (const Instruction &I : instructions(F))
      I.AuxData = 0;
    F->InstAuxDataMax = 0;
  }

  uint32_t getMax() const { return F->InstAuxDataMax; }

  uint32_t getRaw(const Instruction &I) const {
    assert(I.getFunction() == F);
    return I.AuxData;
  }

  uint32_t &getRaw(const Instruction &I) {
    assert(I.getFunction() == F);
    return I.AuxData;
  }

  void release(uint32_t MaxValue = UINT32_MAX) {
    if (!F)
      return;
    F->InstAuxDataMax = MaxValue;
    for (const Instruction &I : instructions(F))
      assert(I.AuxData <= F->InstAuxDataMax && "release() with wrong MaxValue");
    F->InstAuxDataUsed = false;
    F = nullptr;
  }
};

/// Map from Instruction to an uint32_t. Starts uninitialized.
class InstructionInlineData : public InstructionInlineStorage {
public:
  InstructionInlineData(const Function *F = nullptr)
      : InstructionInlineStorage(F) {}

  using InstructionInlineStorage::clearToZero;
  using InstructionInlineStorage::release;

  uint32_t operator[](const Instruction &I) const { return getRaw(I); }
  uint32_t &operator[](const Instruction &I) { return getRaw(I); }
};

/// Set of instructions in a function, stored inline in the instruction.
class InstructionInlineSet : public InstructionInlineStorage {
public:
  uint32_t SetValue;

  InstructionInlineSet(const Function *F = nullptr)
      : InstructionInlineStorage(F) {
    if (getMax() == UINT32_MAX)
      clearToZero();
    SetValue = getMax() + 1;
  }
  ~InstructionInlineSet() { release(SetValue); }

  bool contains(const Instruction &I) {
    assert(I.getFunction() == F);
    return getRaw(I) <= SetValue;
  }
  bool insert(const Instruction &I) {
    bool Contained = contains(I);
    getRaw(I) = SetValue + 1;
    return !Contained;
  }
  void erase(const Instruction &I) {
    assert(I.getFunction() == F);
    getRaw(I) = 0;
  }
};

} // namespace llvm

#endif
