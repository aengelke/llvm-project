//===--- DwarfCFIEHPrinter.h - DWARF-based Unwind Information Printer -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_DWARFCFIEHPRINTER_H
#define LLVM_TOOLS_LLVM_READOBJ_DWARFCFIEHPRINTER_H

#include "llvm-readobj.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFCFIPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/DebugInfo/DWARF/DWARFUnwindTablePrinter.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/type_traits.h"

namespace llvm {
namespace DwarfCFIEH {

template <typename ELFT> class PrinterContext {
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Phdr = typename ELFT::Phdr;

  ScopedPrinter &W;
  const object::ELFObjectFile<ELFT> &ObjF;

  void printEHFrameHdr(const Elf_Phdr *EHFramePHdr) const;
  void printEHFrame(const Elf_Shdr *EHFrameShdr) const;

public:
  PrinterContext(ScopedPrinter &W, const object::ELFObjectFile<ELFT> &ObjF)
      : W(W), ObjF(ObjF) {}

  void printUnwindInformation() const;
};

template <class ELFT>
static const typename ELFT::Shdr *
findSectionByAddress(const object::ELFObjectFile<ELFT> &ObjF, uint64_t Addr) {
  Expected<typename ELFT::ShdrRange> SectionsOrErr =
      ObjF.getELFFile().sections();
  if (!SectionsOrErr)
    reportError(SectionsOrErr.takeError(), ObjF.getFileName());

  for (const typename ELFT::Shdr &Shdr : *SectionsOrErr)
    if (Shdr.sh_addr == Addr)
      return &Shdr;
  return nullptr;
}

template <typename ELFT>
void PrinterContext<ELFT>::printUnwindInformation() const {
  const object::ELFFile<ELFT> &Obj = ObjF.getELFFile();

  Expected<typename ELFT::PhdrRange> PhdrsOrErr = Obj.program_headers();
  if (!PhdrsOrErr)
    reportError(PhdrsOrErr.takeError(), ObjF.getFileName());

  for (const Elf_Phdr &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != ELF::PT_GNU_EH_FRAME)
      continue;

    if (Phdr.p_memsz != Phdr.p_filesz)
      reportError(object::createError(
                      "p_memsz does not match p_filesz for GNU_EH_FRAME"),
                  ObjF.getFileName());
    printEHFrameHdr(&Phdr);
    break;
  }

  Expected<typename ELFT::ShdrRange> SectionsOrErr = Obj.sections();
  if (!SectionsOrErr)
    reportError(SectionsOrErr.takeError(), ObjF.getFileName());

  for (const Elf_Shdr &Shdr : *SectionsOrErr) {
    Expected<StringRef> NameOrErr = Obj.getSectionName(Shdr);
    if (!NameOrErr)
      reportError(NameOrErr.takeError(), ObjF.getFileName());
    if (*NameOrErr == ".eh_frame")
      printEHFrame(&Shdr);
  }
}

static Expected<dwarf::UnwindTable>
createX86_64CompactUnwindTable(ArrayRef<std::pair<uint64_t, uint64_t>> Descs) {
  using namespace llvm::dwarf;
  UnwindRow NullRow;
  NullRow.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, 8);
  NullRow.getRegisterLocations().setRegisterLocation(
      16, UnwindLocation::createAtCFAPlusOffset(-8));

  UnwindTable::RowContainer Rows;

  static constexpr uint8_t RegSaveOrder[] = {
      // X86::RBP, X86::R15, X86::R14, X86::R13, X86::R12, X86::RBX,
      6, 15, 14, 13, 12, 6, 3,
  };

  for (auto [Idx, Entry] : enumerate(Descs.drop_back())) {
    uint64_t Loc = Entry.first;
    uint64_t Len = Descs[Idx + 1].first - Loc;
    uint64_t Desc = Entry.second;
    unsigned Mode = (Desc >> 29) & 0x7;
    unsigned PrologueSize = (Desc >> 32) & 0xff;
    unsigned EpilogueSize = (Desc >> 40) & 0xff;
    if (Len < PrologueSize + EpilogueSize)
      return createStringError(errc::invalid_argument,
                               "CU len smaller than prologue+epilogue");
    switch (Mode) {
    case 7: // DWARF
      // TODO: read FDE here?
      break;
    case 0: // NULL
      NullRow.setAddress(Loc);
      Rows.push_back(NullRow);
      break;
    case 1: { // RBP
      unsigned InnerPrologueSize = (Desc >> 18) & 0xff;
      uint64_t MovRbpRspLoc = Loc;
      UnwindRow Row = NullRow;
      if (PrologueSize > InnerPrologueSize + 3) {
        Row.setAddress(Loc);
        Rows.push_back(Row);
        MovRbpRspLoc = Loc + PrologueSize - InnerPrologueSize - 3;
      }
      Row.setAddress(MovRbpRspLoc);
      Row.getRegisterLocations().setRegisterLocation(
          6, UnwindLocation::createAtCFAPlusOffset(-16));
      Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, 16);
      if (PrologueSize > InnerPrologueSize)
        Rows.push_back(Row);
      Row.setAddress(Loc + PrologueSize - InnerPrologueSize);
      Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(6, 16);
      if (PrologueSize > 0 && PrologueSize >= InnerPrologueSize)
        Rows.push_back(Row);
      unsigned SaveOff = 8;
      for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
        if (!(Desc & (1 << Idx)))
          continue;
        SaveOff += 8;
        Row.getRegisterLocations().setRegisterLocation(
            Reg, UnwindLocation::createAtCFAPlusOffset(-SaveOff));
      }
      if (PrologueSize == 0 || PrologueSize < InnerPrologueSize ||
          SaveOff != 16) {
        Row.setAddress(Loc + PrologueSize);
        Rows.push_back(Row);
      }
      if (EpilogueSize > 0) {
        NullRow.setAddress(Loc + Len - EpilogueSize);
        Rows.push_back(NullRow);
      }
      break;
    }
    case 2: { // RSP
      unsigned FrameSize = 8 * ((Desc >> 7) & 0x7ffff);
      unsigned SavedRegs = 0;
      unsigned NaturalPrologueSize = 0;
      for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
        if (!(Desc & (1 << Idx)))
          continue;
        if (SavedRegs > 0)
          NaturalPrologueSize += Reg < 8 ? 1 : 2;
        SavedRegs += 1;
      }
      unsigned SubRspDelta = FrameSize - 8 * (SavedRegs + 1);
      unsigned SubRspSize = SubRspDelta == 0    ? 0
                            : SubRspDelta == 8  ? 1
                            : SubRspDelta < 128 ? 4
                                                : 7;
      if (SavedRegs > 0)
        NaturalPrologueSize += SubRspSize;

      DEBUG_WITH_TYPE("compact-unwind",
                      dbgs() << "Loc=" << Loc << " FrameSize=" << FrameSize
                             << " NPS=" << NaturalPrologueSize
                             << " PS=" << PrologueSize << " SR=" << SavedRegs
                             << " SRS=" << SubRspSize << "\n");

      UnwindRow Row = NullRow;
      Row.setAddress(Loc);
      if (PrologueSize > NaturalPrologueSize)
        Rows.push_back(Row);
      uint64_t PrologueLoc = Loc + PrologueSize - NaturalPrologueSize;
      unsigned SaveOff = 8;
      for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
        if (!(Desc & (1 << Idx)))
          continue;
        if (SaveOff > 8)
          PrologueLoc += Reg < 8 ? 1 : 2;
        SaveOff += 8;
        Row.getCFAValue() =
            UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
        Row.getRegisterLocations().setRegisterLocation(
            Reg, UnwindLocation::createAtCFAPlusOffset(-SaveOff));
        Row.setAddress(PrologueLoc);
        if (int64_t(PrologueLoc) >= int64_t(Loc))
          Rows.push_back(Row);
      }
      if (SubRspSize != 0 || PrologueLoc + SubRspSize != Loc + PrologueSize) {
        Row.setAddress(Loc + PrologueSize);
        Row.getCFAValue() =
            UnwindLocation::createIsRegisterPlusOffset(7, FrameSize);
        Rows.push_back(Row);
      }
      if (EpilogueSize > 0) {
        uint64_t EpilogueLoc = Loc + Len - EpilogueSize;
        bool First = true;
        DEBUG_WITH_TYPE("compact-unwind",
                        dbgs() << "ELoc=" << EpilogueLoc
                               << " ES=" << EpilogueSize << " SR=" << SavedRegs
                               << " SRS=" << SubRspSize << "\n");
        if (SubRspSize > 0) {
          Row.getCFAValue() =
              UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
          Row.setAddress(EpilogueLoc);
          Rows.push_back(Row);
          First = false;
        }
        for (auto [Idx, Reg] : enumerate(reverse(RegSaveOrder))) {
          if (!(Desc & (1 << (5 - Idx))))
            continue;
          if (!First)
            EpilogueLoc += Reg < 8 ? 1 : 2;
          First = false;
          SaveOff -= 8;
          Row.getCFAValue() =
              UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
          Row.getRegisterLocations().removeRegisterLocation(Reg);
          Row.setAddress(EpilogueLoc);
          if (EpilogueLoc >= Loc + Len)
            break;
          Rows.push_back(Row);
        }
      }
      break;
    }
    default:
      return createStringError(errc::invalid_argument,
                               "unsupported compact unwind mode %d", Mode);
    }
  }
  return UnwindTable(std::move(Rows));
}

template <typename ELFT>
void PrinterContext<ELFT>::printEHFrameHdr(const Elf_Phdr *EHFramePHdr) const {
  DictScope L(W, "EHFrameHeader");
  uint64_t EHFrameHdrAddress = EHFramePHdr->p_vaddr;
  W.startLine() << format("Address: 0x%" PRIx64 "\n", EHFrameHdrAddress);
  W.startLine() << format("Offset: 0x%" PRIx64 "\n", (uint64_t)EHFramePHdr->p_offset);
  W.startLine() << format("Size: 0x%" PRIx64 "\n", (uint64_t)EHFramePHdr->p_memsz);

  const object::ELFFile<ELFT> &Obj = ObjF.getELFFile();
  if (const Elf_Shdr *EHFrameHdr =
          findSectionByAddress(ObjF, EHFramePHdr->p_vaddr)) {
    Expected<StringRef> NameOrErr = Obj.getSectionName(*EHFrameHdr);
    if (!NameOrErr)
      reportError(NameOrErr.takeError(), ObjF.getFileName());
    W.printString("Corresponding Section", *NameOrErr);
  }

  Expected<ArrayRef<uint8_t>> Content = Obj.getSegmentContents(*EHFramePHdr);
  if (!Content)
    reportError(Content.takeError(), ObjF.getFileName());

  DataExtractor DE(*Content, ELFT::Endianness == llvm::endianness::little);

  DictScope D(W, "Header");
  uint64_t Offset = 0;

  auto Version = DE.getU8(&Offset);
  W.printNumber("version", Version);
  if (Version == 2) {
    // Collect pairs of address and descriptor for printing the unwind table.
    SmallVector<std::pair<uint64_t, uint64_t>> Descs;

    uint64_t PtrEnc = DE.getU8(&Offset);
    W.startLine() << format("ptr_enc: 0x%" PRIx64 "\n", PtrEnc);
    if (PtrEnc != (dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4))
      reportError(object::createError("unexpected encoding ptr_enc"),
                  ObjF.getFileName());
    DE.getU16(&Offset); // Padding.
    uint32_t PersonalitiesOff = DE.getU32(&Offset);
    uint32_t GlobalDescsOff = DE.getU32(&Offset);
    uint32_t PageCount = DE.getU32(&Offset);
    for (uint32_t i = 0; i != PageCount + 1; ++i) {
      dbgs() << "Page " << i << "/" << PageCount << "\n";
      uint32_t Pc = EHFrameHdrAddress + DE.getU32(&Offset);
      uint64_t PageStart = DE.getU32(&Offset);
      uint32_t FirstLSDAOff = DE.getU32(&Offset);
      // Print entry even for the sentinel page.
      DictScope D(W, std::string("page ") + std::to_string(i));
      W.startLine() << format("pc: 0x%" PRIx64 "\n", Pc);

      uint64_t PageOff = PageStart;
      unsigned EntryCount = DE.getU16(&PageOff);
      uint64_t LocalDescsOff = PageStart + DE.getU16(&PageOff);
      dbgs() << "Page " << i << "/" << PageCount << " " << EntryCount << " "
             << PageOff << "\n";

      if (i == PageCount) {
        if (EntryCount != 0)
          reportError(object::createError("sentinel CU page must be empty"),
                      ObjF.getFileName());
        Descs.emplace_back(Pc, 0);
        break;
      }

      for (unsigned j = 0; j != EntryCount; ++j) {
        uint32_t Entry = DE.getU32(&PageOff);
        uint32_t DescIdx = Entry & 0xfff;
        uint64_t DescOff;
        if (DescIdx < 0x1000 - 341)
          DescOff = GlobalDescsOff + sizeof(uint64_t) * DescIdx;
        else
          DescOff = LocalDescsOff + sizeof(uint64_t) * (DescIdx - 0x1000 + 341);
        uint64_t Desc = DE.getU64(&DescOff);
        // XXX: print personality function, if any?
        if (((Desc >> 29) & 7) == 7) {
          uint64_t FDEAddr = EHFrameHdrAddress + (Desc & 0x1fffffff);
          W.startLine() << format("entry: 0x%" PRIx64 " %016" PRIx64
                                  " (FDE 0x%" PRIx64 ")\n",
                                  Pc + (Entry >> 12), Desc, FDEAddr);
        } else {
          W.startLine() << format("entry: 0x%" PRIx64 " %016" PRIx64 "\n",
                                  Pc + (Entry >> 12), Desc);
        }
        Descs.emplace_back(Pc + (Entry >> 12), Desc);
      }
    }
    // XXX: LSDA table

    // Construct unwind table from descriptors.
    //
    if (Expected<dwarf::UnwindTable> RowsOrErr =
            createX86_64CompactUnwindTable(Descs))
      printUnwindTable(*RowsOrErr, W.getOStream(), DIDumpOptions{}, 1);
    else
      reportError(RowsOrErr.takeError(), ObjF.getFileName());
    return;
  }

  if (Version != 1)
    reportError(
        object::createError("only version 1 of .eh_frame_hdr is supported"),
        ObjF.getFileName());

  uint64_t EHFramePtrEnc = DE.getU8(&Offset);
  W.startLine() << format("eh_frame_ptr_enc: 0x%" PRIx64 "\n", EHFramePtrEnc);
  unsigned EHFramePtrSize = 0;
  if (EHFramePtrEnc == (dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4))
    EHFramePtrSize = 4;
  else if (EHFramePtrEnc == (dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8))
    EHFramePtrSize = 8;
  else
    reportError(object::createError("unexpected encoding eh_frame_ptr_enc"),
                ObjF.getFileName());

  uint64_t FDECountEnc = DE.getU8(&Offset);
  W.startLine() << format("fde_count_enc: 0x%" PRIx64 "\n", FDECountEnc);
  if (FDECountEnc != dwarf::DW_EH_PE_udata4)
    reportError(object::createError("unexpected encoding fde_count_enc"),
                ObjF.getFileName());

  uint64_t TableEnc = DE.getU8(&Offset);
  W.startLine() << format("table_enc: 0x%" PRIx64 "\n", TableEnc);
  unsigned TableEntrySize = 0;
  if (TableEnc == (dwarf::DW_EH_PE_datarel | dwarf::DW_EH_PE_sdata4))
    TableEntrySize = 4;
  else if (TableEnc == (dwarf::DW_EH_PE_datarel | dwarf::DW_EH_PE_sdata8))
    TableEntrySize = 8;
  else
    reportError(object::createError("unexpected encoding table_enc 0x" +
                                    Twine::utohexstr(TableEnc)),
                ObjF.getFileName());

  auto EHFramePtr =
      DE.getSigned(&Offset, EHFramePtrSize) + EHFrameHdrAddress + 4;
  W.startLine() << format("eh_frame_ptr: 0x%" PRIx64 "\n", EHFramePtr);

  auto FDECount = DE.getUnsigned(&Offset, 4);
  W.printNumber("fde_count", FDECount);

  unsigned NumEntries = 0;
  uint64_t PrevPC = 0;
  while (Offset + 2 * TableEntrySize <= EHFramePHdr->p_memsz &&
         NumEntries < FDECount) {
    DictScope D(W, std::string("entry ") + std::to_string(NumEntries));

    auto InitialPC = DE.getSigned(&Offset, TableEntrySize) + EHFrameHdrAddress;
    W.startLine() << format("initial_location: 0x%" PRIx64 "\n", InitialPC);
    auto Address = DE.getSigned(&Offset, TableEntrySize) + EHFrameHdrAddress;
    W.startLine() << format("address: 0x%" PRIx64 "\n", Address);

    if (InitialPC < PrevPC)
      reportError(object::createError("initial_location is out of order"),
                  ObjF.getFileName());

    PrevPC = InitialPC;
    ++NumEntries;
  }
}

template <typename ELFT>
void PrinterContext<ELFT>::printEHFrame(const Elf_Shdr *EHFrameShdr) const {
  uint64_t Address = EHFrameShdr->sh_addr;
  uint64_t ShOffset = EHFrameShdr->sh_offset;
  W.startLine() << format(".eh_frame section at offset 0x%" PRIx64
                          " address 0x%" PRIx64 ":\n",
                          ShOffset, Address);
  W.indent();

  Expected<ArrayRef<uint8_t>> DataOrErr =
      ObjF.getELFFile().getSectionContents(*EHFrameShdr);
  if (!DataOrErr)
    reportError(DataOrErr.takeError(), ObjF.getFileName());

  // Construct DWARFDataExtractor to handle relocations ("PC Begin" fields).
  std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(
      ObjF, DWARFContext::ProcessDebugRelocations::Process, nullptr);
  DWARFDataExtractor DE(
      DICtx->getDWARFObj(), DICtx->getDWARFObj().getEHFrameSection(),
      ELFT::Endianness == llvm::endianness::little, ELFT::Is64Bits ? 8 : 4);
  DWARFDebugFrame EHFrame(Triple::ArchType(ObjF.getArch()), /*IsEH=*/true,
                          /*EHFrameAddress=*/Address);
  if (Error E = EHFrame.parse(DE))
    reportError(std::move(E), ObjF.getFileName());

  for (const dwarf::FrameEntry &Entry : EHFrame) {
    std::optional<uint64_t> InitialLocation;
    if (const dwarf::CIE *CIE = dyn_cast<dwarf::CIE>(&Entry)) {
      W.startLine() << format("[0x%" PRIx64 "] CIE length=%" PRIu64 "\n",
                              Address + CIE->getOffset(), CIE->getLength());
      W.indent();

      W.printNumber("version", CIE->getVersion());
      W.printString("augmentation", CIE->getAugmentationString());
      W.printNumber("code_alignment_factor", CIE->getCodeAlignmentFactor());
      W.printNumber("data_alignment_factor", CIE->getDataAlignmentFactor());
      W.printNumber("return_address_register", CIE->getReturnAddressRegister());
    } else {
      const dwarf::FDE *FDE = cast<dwarf::FDE>(&Entry);
      W.startLine() << format("[0x%" PRIx64 "] FDE length=%" PRIu64
                              " cie=[0x%" PRIx64 "]\n",
                              Address + FDE->getOffset(), FDE->getLength(),
                              Address + FDE->getLinkedCIE()->getOffset());
      W.indent();

      InitialLocation = FDE->getInitialLocation();
      W.startLine() << format("initial_location: 0x%" PRIx64 "\n",
                              *InitialLocation);
      W.startLine() << format(
          "address_range: 0x%" PRIx64 " (end : 0x%" PRIx64 ")\n",
          FDE->getAddressRange(),
          FDE->getInitialLocation() + FDE->getAddressRange());
    }

    W.getOStream() << "\n";
    W.startLine() << "Program:\n";
    W.indent();
    auto DumpOpts = DIDumpOptions();
    DumpOpts.IsEH = true;
    printCFIProgram(Entry.cfis(), W.getOStream(), DumpOpts, W.getIndentLevel(),
                    InitialLocation);
    W.unindent();
    W.unindent();
    W.getOStream() << "\n";
  }

  W.unindent();
}
} // namespace DwarfCFIEH
} // namespace llvm

#endif
