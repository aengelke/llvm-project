//===- llvm/ADT/SlabVectorStorage.h - Storage for many vectors --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SLABVECTORSTORAGE_H
#define LLVM_ADT_SLABVECTORSTORAGE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <type_traits>

namespace llvm {

template <class T> class VectorWriter;

/// Memory-efficient storage for many small vectors, optimized for append-only.
template <class T> class SlabVectorStorage {
  friend VectorWriter<T>;

  static_assert(std::is_trivially_copy_constructible<T>::value &&
                    std::is_trivially_move_constructible<T>::value &&
                    std::is_trivially_destructible<T>::value,
                "support for non-POD types not implemented");

  /// Slabs of small storage.
  SmallVector<std::unique_ptr<T[]>> Slabs;
  /// Vectors that grow larger. SmallVector<T, 0> so that a grow of the outer
  /// vector doesn't invalidate pointers into the inner vector.
  SmallVector<SmallVector<T, 0>, 0> Vectors;

  /// Pointer to end of last element in last slab.
  T *Cur;
  /// End of last slab.
  T *End;
  /// Slab size
  unsigned SlabSize;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// Whether there is currently an active VectorWriter. There must be at most
  /// one writer at a time.
  bool HasWriter = false;
#endif

public:
  SlabVectorStorage(unsigned SlabSize = 1024)
      : Cur(nullptr), End(nullptr), SlabSize(SlabSize) {}

  /// Reference to a vector in the SlabVectorStorage.
  class Vector {
    friend VectorWriter<T>;

    /// Begin pointer
    T *Begin;
    /// Number of elements
    unsigned Size;
    /// If non-negative: vector capacity; if negative: data is backed by a
    /// SmallVector in Vectors[-1 - CapIdx].
    int CapIdx;

  public:
    Vector() : Begin(nullptr), Size(0), CapIdx(0) {}

    T *begin() const { return Begin; }
    T *end() const { return Begin + Size; }

    size_t size() const { return Size; }

    void clear() { Size = 0; }

    operator ArrayRef<T>() const { return ArrayRef(Begin, Size); }
    operator MutableArrayRef<T>() { return MutableArrayRef(Begin, Size); }
  };

  /// Clear all associated data. Vector references are *not* updated.
  void clear() {
    Slabs.clear();
    Vectors.clear();
    Cur = End = nullptr;
  }

private:
  void allocSlab() {
    Cur = Slabs.emplace_back(new T[SlabSize]).get();
    End = Cur + SlabSize;
  }
};

template <class T> class VectorWriter {
  SmallVectorImpl<T> *SV = nullptr;
  SlabVectorStorage<T> *const VS = nullptr;
  typename SlabVectorStorage<T>::Vector *const VSV = nullptr;
  T *Cur;
  T *End;

  static_assert(std::is_trivially_copy_constructible<T>::value &&
                    std::is_trivially_move_constructible<T>::value &&
                    std::is_trivially_destructible<T>::value,
                "support for non-POD types not implemented");

public:
  /// Write to a SmallVector. While the writer exists, the size is temporarily
  /// grown to its capacity and internal pointers are stored. No simultaneous
  /// use is allowed.
  VectorWriter(SmallVectorImpl<T> &SVP) : SV(&SVP) { updateFromVector(); }

  VectorWriter(SlabVectorStorage<T> &VSP,
               typename SlabVectorStorage<T>::Vector &VSVP)
      : VS(&VSP), VSV(&VSVP) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(!VS->HasWriter);
    VS->HasWriter = true;
#endif
    if (VSV->CapIdx < 0) {
      SV = &VS->Vectors[-1 - VSV->CapIdx];
      updateFromVector();
      return;
    }

    if (!VSV->begin()) {
      VSV->Begin = Cur = VS->Cur;
      End = VS->End;
    } else {
      Cur = VSV->end();
      End = VSV->begin() + VSV->CapIdx;
    }
  }

  VectorWriter(const VectorWriter &) = delete;
  VectorWriter(VectorWriter &&) = delete;

  VectorWriter &operator=(const VectorWriter &) = delete;
  VectorWriter &operator=(VectorWriter &&) = delete;

  ~VectorWriter() {
    // If there's a SmallVector, shrink it back to the actual size.
    if (SV)
      SV->truncate(size());
    if (VSV) {
      if (End == VS->End) {
        VSV->CapIdx = size();
        VS->Cur = Cur;
      }
      VSV->Size = size();
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
      VS->HasWriter = false;
#endif
    }
  }

  T *begin() const { return SV ? SV->begin() : VSV->begin(); }
  T *end() const { return Cur; }
  size_t size() const { return end() - begin(); }

  void clear() { Cur = begin(); }

  void push_back(T Elt) {
    if (LLVM_UNLIKELY(Cur == End))
      grow(1);
    memcpy(reinterpret_cast<void *>(Cur), &Elt, sizeof(T));
    Cur++;
  }

  void append(ArrayRef<T> Elts) {
    if (Elts.empty())
      return;
    if (LLVM_UNLIKELY(Cur + Elts.size() > End))
      grow(Elts.size());
    memcpy(reinterpret_cast<void *>(Cur), Elts.data(), Elts.size() * sizeof(T));
    Cur += Elts.size();
  }

private:
  /// Update internal pointers to the small vector, e.g. after a resize.
  void updateFromVector() {
    Cur = SV->end();
    SV->resize_for_overwrite(SV->capacity());
    End = SV->end();
    if (VSV)
      VSV->Begin = SV->begin();
  }

  void grow(size_t GrowSize);
};

template <class T> void VectorWriter<T>::grow(size_t GrowSize) {
  // If we have a small vector, grow it and update internal pointers.
  if (SV) {
    SV->reserve(SV->size() + GrowSize);
    updateFromVector();
    return;
  }

  // We're at the end of the last slab and can simply grow into empty space.
  if (Cur == VS->Cur && End != VS->End && VS->Cur != VS->End) {
    End = VS->End;
    return;
  }

  size_t Size = size(); // Copy size before we mess with pointers
  // When copying a large amount elements, move to a small vector instead.
  // Chances are that more elements will follow, so a SmallVector is a sensible
  // option.
  if ((Size + GrowSize) * 2 > VS->SlabSize) {
    VSV->CapIdx = -1 - VS->Vectors.size();
    SV = &VS->Vectors.emplace_back();
    SV->reserve((Size + GrowSize) * 2);
    SV->append(VSV->begin(), Cur); // can't use begin(), because SV is now set.
    updateFromVector();
    return;
  }

  // Grow inside slab-allocated memory by moving the allocation to the end
  // We may need to allocate a slab for that, first.
  if (VS->Cur + (Size + GrowSize) > VS->End)
    VS->allocSlab();
  if (Size)
    memcpy(reinterpret_cast<void *>(VS->Cur), VSV->Begin, sizeof(T) * Size);
  VSV->Begin = VS->Cur;
  Cur = VS->Cur + Size;
  End = VS->End;
}

} // namespace llvm

#endif
