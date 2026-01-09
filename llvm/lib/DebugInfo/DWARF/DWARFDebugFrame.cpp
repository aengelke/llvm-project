//===- DWARFDebugFrame.h - Parsing of .debug_frame ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCFIPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFUnwindTablePrinter.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFCFIProgram.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace dwarf;

Expected<UnwindTable> llvm::dwarf::createUnwindTable(const FDE *Fde) {
  const CIE *Cie = Fde->getLinkedCIE();
  if (Cie == nullptr)
    return createStringError(errc::invalid_argument,
                             "unable to get CIE for FDE at offset 0x%" PRIx64,
                             Fde->getOffset());

  if (!Fde->getCompactUnwind().empty()) {
    UnwindRow NullRow;
    NullRow.setAddress(Fde->getInitialLocation());
    NullRow.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, 8);
    NullRow.getRegisterLocations().setRegisterLocation(16, UnwindLocation::createAtCFAPlusOffset(-8));

    ArrayRef<FDECompactUnwind> CUs = Fde->getCompactUnwind();
    UnwindTable::RowContainer Rows;
    uint64_t Loc = Fde->getInitialLocation();

    if (!CUs.empty() && CUs[0].Skip > 0) {
      Rows.push_back(NullRow);
      Loc += CUs[0].Skip;
    }

    static constexpr uint8_t RegSaveOrder[] = {
      // X86::RBP, X86::R15, X86::R14, X86::R13, X86::R12, X86::RBX,
      6, 15, 14, 13, 12, 3,
    };

    for (const auto *CU = CUs.begin(), *End = CUs.end(); CU != End; ++CU) {
      // Loc += CU->Skip;
      uint64_t Len = CU + 1 == End ? Fde->getInitialLocation() + Fde->getAddressRange() - Loc : CU[1].Skip;
      unsigned Mode = (CU->Desc >> 29) & 0x7;
      unsigned PrologueSize = (CU->Desc >> 32) & 0xff;
      unsigned EpilogueSize = (CU->Desc >> 40) & 0xff;
      if (Len < PrologueSize + EpilogueSize)
        return createStringError(errc::invalid_argument, "CU len smaller than prologue+epilogue");
      switch (Mode) {
      case 0: // NULL
        NullRow.setAddress(Loc);
        Rows.push_back(NullRow);
        break;
      case 1: { // RBP
        unsigned InnerPrologueSize = (CU->Desc >> 18) & 0xff;
        uint64_t MovRbpRspLoc = Loc;
        UnwindRow Row = NullRow;
        if (PrologueSize > InnerPrologueSize + 3) {
          Row.setAddress(Loc);
          Rows.push_back(Row);
          MovRbpRspLoc = Loc + PrologueSize - InnerPrologueSize - 3;
        }
        Row.setAddress(MovRbpRspLoc);
        Row.getRegisterLocations().setRegisterLocation(6, UnwindLocation::createAtCFAPlusOffset(-16));
        Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, 16);
        if (PrologueSize > InnerPrologueSize)
          Rows.push_back(Row);
        Row.setAddress(Loc + PrologueSize - InnerPrologueSize);
        Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(6, 16);
        if (PrologueSize > 0 && PrologueSize >= InnerPrologueSize)
          Rows.push_back(Row);
        unsigned SaveOff = 8;
        for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
          if (!(CU->Desc & (1 << Idx)))
            continue;
          SaveOff += 8;
          Row.getRegisterLocations().setRegisterLocation(Reg, UnwindLocation::createAtCFAPlusOffset(-SaveOff));
        }
        if (PrologueSize == 0 || PrologueSize < InnerPrologueSize || SaveOff != 16) {
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
        unsigned FrameSize = 8 * ((CU->Desc >> 6) & 0xfffff);
        unsigned SavedRegs = 0;
        unsigned NaturalPrologueSize = 0;
        for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
          if (!(CU->Desc & (1 << Idx)))
            continue;
          if (SavedRegs > 0)
            NaturalPrologueSize += Reg < 8 ? 1 : 2;
          SavedRegs += 1;
        }
        unsigned SubRspDelta = FrameSize - 8 * (SavedRegs + 1);
        unsigned SubRspSize = SubRspDelta == 0 ? 0 : SubRspDelta == 8 ? 1 : SubRspDelta < 128 ? 4 : 7;
        NaturalPrologueSize += SubRspSize;

        dbgs() << "Loc=" << Loc << " FrameSize=" << FrameSize << " NPS=" << NaturalPrologueSize
          << " PS=" << PrologueSize << " SR=" << SavedRegs << " SRS=" << SubRspSize << "\n";

        UnwindRow Row = NullRow;
        Row.setAddress(Loc);
        if (PrologueSize > NaturalPrologueSize)
          Rows.push_back(Row);
        uint64_t PrologueLoc = Loc + PrologueSize - NaturalPrologueSize;
        // Row.setAddress(Loc);
        // Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, FrameSize);
        // Row.getRegisterLocations().setRegisterLocation(16, UnwindLocation::createAtCFAPlusOffset(-8));
        unsigned SaveOff = 8;
        for (auto [Idx, Reg] : enumerate(RegSaveOrder)) {
          if (!(CU->Desc & (1 << Idx)))
            continue;
          if (SaveOff > 8)
            PrologueLoc += Reg < 8 ? 1 : 2;
          SaveOff += 8;
          Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
          Row.getRegisterLocations().setRegisterLocation(Reg, UnwindLocation::createAtCFAPlusOffset(-SaveOff));
          Row.setAddress(PrologueLoc);
          if (int64_t(PrologueLoc) >= int64_t(Loc))
            Rows.push_back(Row);
        }
        if (SubRspSize != 0 || PrologueLoc + SubRspSize != Loc + PrologueSize) {
          Row.setAddress(Loc + PrologueSize);
          Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, FrameSize);
          Rows.push_back(Row);
        }
        if (EpilogueSize > 0) {
          uint64_t EpilogueLoc = Loc + Len - EpilogueSize;
          bool First = true;
          // dbgs() << "ELoc=" << EpilogueLoc
          //   << " ES=" << EpilogueSize << " SR=" << SavedRegs << " SRS=" << SubRspSize << "\n";
          if (SubRspSize > 0) {
            Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
            Row.setAddress(EpilogueLoc);
            Rows.push_back(Row);
            First = false;
          }
          for (auto [Idx, Reg] : enumerate(reverse(RegSaveOrder))) {
            if (!(CU->Desc & (1 << (5 - Idx))))
              continue;
            if (!First)
              EpilogueLoc += Reg < 8 ? 1 : 2;
            // dbgs() << "Idx=" << Idx << " Reg=" << Reg << " ELoc=" << EpilogueLoc
            //   << " ES=" << EpilogueSize << " SO=" << SaveOff << " SRS=" << SubRspSize << " L+L" << (Loc+Len) << "\n";
            First = false;
            SaveOff -= 8;
            Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(7, SaveOff);
            // Row.getRegisterLocations().removeRegisterLocation(Reg);
            Row.setAddress(EpilogueLoc);
            if (EpilogueLoc >= Loc + Len)
              break;
            Rows.push_back(Row);
          }
        }
        break;
      }
      default:
        return createStringError(errc::invalid_argument, "unsupported compact unwind mode %d", Mode);
      }
      Loc += Len;
    }
    // Rows.push_back(NullRow);
    return UnwindTable(std::move(Rows));
  }

  // Rows will be empty if there are no CFI instructions.
  if (Cie->cfis().empty() && Fde->cfis().empty())
    return UnwindTable({});

  UnwindTable::RowContainer CieRows;
  UnwindRow Row;
  Row.setAddress(Fde->getInitialLocation());
  if (Error CieError = parseRows(Cie->cfis(), Row, nullptr).moveInto(CieRows))
    return std::move(CieError);
  // We need to save the initial locations of registers from the CIE parsing
  // in case we run into DW_CFA_restore or DW_CFA_restore_extended opcodes.
  UnwindTable::RowContainer FdeRows;
  const RegisterLocations InitialLocs = Row.getRegisterLocations();
  if (Error FdeError =
          parseRows(Fde->cfis(), Row, &InitialLocs).moveInto(FdeRows))
    return std::move(FdeError);

  UnwindTable::RowContainer AllRows;
  AllRows.insert(AllRows.end(), CieRows.begin(), CieRows.end());
  AllRows.insert(AllRows.end(), FdeRows.begin(), FdeRows.end());

  // May be all the CFI instructions were DW_CFA_nop amd Row becomes empty.
  // Do not add that to the unwind table.
  if (Row.getRegisterLocations().hasLocations() ||
      Row.getCFAValue().getLocation() != UnwindLocation::Unspecified)
    AllRows.push_back(Row);
  return UnwindTable(std::move(AllRows));
}

Expected<UnwindTable> llvm::dwarf::createUnwindTable(const CIE *Cie) {
  // Rows will be empty if there are no CFI instructions.
  if (Cie->cfis().empty())
    return UnwindTable({});

  UnwindTable::RowContainer Rows;
  UnwindRow Row;
  if (Error CieError = parseRows(Cie->cfis(), Row, nullptr).moveInto(Rows))
    return std::move(CieError);
  // May be all the CFI instructions were DW_CFA_nop amd Row becomes empty.
  // Do not add that to the unwind table.
  if (Row.getRegisterLocations().hasLocations() ||
      Row.getCFAValue().getLocation() != UnwindLocation::Unspecified)
    Rows.push_back(Row);
  return UnwindTable(std::move(Rows));
}

// Returns the CIE identifier to be used by the requested format.
// CIE ids for .debug_frame sections are defined in Section 7.24 of DWARFv5.
// For CIE ID in .eh_frame sections see
// https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
constexpr uint64_t getCIEId(bool IsDWARF64, bool IsEH) {
  if (IsEH)
    return 0;
  if (IsDWARF64)
    return DW64_CIE_ID;
  return DW_CIE_ID;
}

void CIE::dump(raw_ostream &OS, DIDumpOptions DumpOpts) const {
  // A CIE with a zero length is a terminator entry in the .eh_frame section.
  if (DumpOpts.IsEH && Length == 0) {
    OS << format("%08" PRIx64, Offset) << " ZERO terminator\n";
    return;
  }

  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !DumpOpts.IsEH ? 16 : 8,
               getCIEId(IsDWARF64, DumpOpts.IsEH))
     << " CIE\n"
     << "  Format:                " << FormatString(IsDWARF64) << "\n";
  if (DumpOpts.IsEH && Version != 1)
    OS << "WARNING: unsupported CIE version\n";
  OS << format("  Version:               %d\n", Version)
     << "  Augmentation:          \"" << Augmentation << "\"\n";
  if (Version >= 4) {
    OS << format("  Address size:          %u\n", (uint32_t)AddressSize);
    OS << format("  Segment desc size:     %u\n",
                 (uint32_t)SegmentDescriptorSize);
  }
  OS << format("  Code alignment factor: %u\n", (uint32_t)CodeAlignmentFactor);
  OS << format("  Data alignment factor: %d\n", (int32_t)DataAlignmentFactor);
  OS << format("  Return address column: %d\n", (int32_t)ReturnAddressRegister);
  if (Personality)
    OS << format("  Personality Address: %016" PRIx64 "\n", *Personality);
  if (!AugmentationData.empty()) {
    OS << "  Augmentation data:    ";
    for (uint8_t Byte : AugmentationData)
      OS << ' ' << hexdigit(Byte >> 4) << hexdigit(Byte & 0xf);
    OS << "\n";
  }
  OS << "\n";
  printCFIProgram(CFIs, OS, DumpOpts, /*IndentLevel=*/1,
                  /*InitialLocation=*/{});
  OS << "\n";

  if (Expected<UnwindTable> RowsOrErr = createUnwindTable(this))
    printUnwindTable(*RowsOrErr, OS, DumpOpts, 1);
  else {
    DumpOpts.RecoverableErrorHandler(joinErrors(
        createStringError(errc::invalid_argument,
                          "decoding the CIE opcodes into rows failed"),
        RowsOrErr.takeError()));
  }
  OS << "\n";
}

void FDE::dump(raw_ostream &OS, DIDumpOptions DumpOpts) const {
  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !DumpOpts.IsEH ? 16 : 8, CIEPointer)
     << " FDE cie=";
  if (LinkedCIE)
    OS << format("%08" PRIx64, LinkedCIE->getOffset());
  else
    OS << "<invalid offset>";
  OS << format(" pc=%08" PRIx64 "...%08" PRIx64 "\n", InitialLocation,
               InitialLocation + AddressRange);
  OS << "  Format:       " << FormatString(IsDWARF64) << "\n";
  if (LSDAAddress)
    OS << format("  LSDA Address: %016" PRIx64 "\n", *LSDAAddress);
  if (!CompactUnwind.empty()) {
    uint64_t Loc = InitialLocation;
    for (const FDECompactUnwind &CU : CompactUnwind) {
      Loc += CU.Skip;
      OS << format("  Descriptor: pc=%016" PRIx64 " desc=%016" PRIx64 "\n", Loc, CU.Desc);
    }
  } else {
    printCFIProgram(CFIs, OS, DumpOpts, /*IndentLevel=*/1, InitialLocation);
  }

  OS << "\n";
  if (Expected<UnwindTable> RowsOrErr = createUnwindTable(this))
    printUnwindTable(*RowsOrErr, OS, DumpOpts, 1);
  else {
    DumpOpts.RecoverableErrorHandler(joinErrors(
        createStringError(errc::invalid_argument,
                          "decoding the FDE opcodes into rows failed"),
        RowsOrErr.takeError()));
  }

  OS << "\n";
}

DWARFDebugFrame::DWARFDebugFrame(Triple::ArchType Arch,
    bool IsEH, uint64_t EHFrameAddress)
    : Arch(Arch), IsEH(IsEH), EHFrameAddress(EHFrameAddress) {}

DWARFDebugFrame::~DWARFDebugFrame() = default;

[[maybe_unused]] static void dumpDataAux(DataExtractor Data, uint64_t Offset,
                                         int Length) {
  errs() << "DUMP: ";
  for (int i = 0; i < Length; ++i) {
    uint8_t c = Data.getU8(&Offset);
    errs().write_hex(c); errs() << " ";
  }
  errs() << "\n";
}

Error DWARFDebugFrame::parse(DWARFDataExtractor Data) {
  uint64_t Offset = 0;
  DenseMap<uint64_t, CIE *> CIEs;

  while (Data.isValidOffset(Offset)) {
    uint64_t StartOffset = Offset;

    uint64_t Length;
    DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);
    bool IsDWARF64 = Format == DWARF64;

    // If the Length is 0, then this CIE is a terminator. We add it because some
    // dumper tools might need it to print something special for such entries
    // (e.g. llvm-objdump --dwarf=frames prints "ZERO terminator").
    if (Length == 0) {
      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, 0, 0, SmallString<8>(), 0, 0, 0, 0, 0,
          SmallString<8>(), 0, 0, std::nullopt, std::nullopt, false, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.push_back(std::move(Cie));
      break;
    }

    // At this point, Offset points to the next field after Length.
    // Length is the structure size excluding itself. Compute an offset one
    // past the end of the structure (needed to know how many instructions to
    // read).
    uint64_t StartStructureOffset = Offset;
    uint64_t EndStructureOffset = Offset + Length;

    // The Id field's size depends on the DWARF format
    Error Err = Error::success();
    uint64_t Id = Data.getRelocatedValue((IsDWARF64 && !IsEH) ? 8 : 4, &Offset,
                                         /*SectionIndex=*/nullptr, &Err);
    if (Err)
      return Err;

    bool IsCIE = Id == getCIEId(IsDWARF64, IsEH);
    bool CompactUnwind = false;
    if (IsCIE) {
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      StringRef AugmentationString(Augmentation ? Augmentation : "");
      uint8_t AddressSize = Version < 4 ? Data.getAddressSize() :
                                          Data.getU8(&Offset);
      Data.setAddressSize(AddressSize);
      uint8_t SegmentDescriptorSize = Version < 4 ? 0 : Data.getU8(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister =
          Version == 1 ? Data.getU8(&Offset) : Data.getULEB128(&Offset);

      // Parse the augmentation data for EH CIEs
      StringRef AugmentationData("");
      uint32_t FDEPointerEncoding = DW_EH_PE_absptr;
      uint32_t LSDAPointerEncoding = DW_EH_PE_omit;
      std::optional<uint64_t> Personality;
      std::optional<uint32_t> PersonalityEncoding;
      if (IsEH) {
        std::optional<uint64_t> AugmentationLength;
        uint64_t StartAugmentationOffset;
        uint64_t EndAugmentationOffset;

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned i = 0, e = AugmentationString.size(); i != e; ++i) {
          switch (AugmentationString[i]) {
          default:
            return createStringError(
                errc::invalid_argument,
                "unknown augmentation character %c in entry at 0x%" PRIx64,
                AugmentationString[i], StartOffset);
          case 'L':
            LSDAPointerEncoding = Data.getU8(&Offset);
            break;
          case 'P': {
            if (Personality)
              return createStringError(
                  errc::invalid_argument,
                  "duplicate personality in entry at 0x%" PRIx64, StartOffset);
            PersonalityEncoding = Data.getU8(&Offset);
            Personality = Data.getEncodedPointer(
                &Offset, *PersonalityEncoding,
                EHFrameAddress ? EHFrameAddress + Offset : 0);
            break;
          }
          case 'R':
            FDEPointerEncoding = Data.getU8(&Offset);
            break;
          case 'S':
            // Current frame is a signal trampoline.
            break;
          case 'z':
            if (i)
              return createStringError(
                  errc::invalid_argument,
                  "'z' must be the first character at 0x%" PRIx64, StartOffset);
            // Parse the augmentation length first.  We only parse it if
            // the string contains a 'z'.
            AugmentationLength = Data.getULEB128(&Offset);
            StartAugmentationOffset = Offset;
            EndAugmentationOffset = Offset + *AugmentationLength;
            break;
          case 'B':
            // B-Key is used for signing functions associated with this
            // augmentation string
            break;
            // This stack frame contains MTE tagged data, so needs to be
            // untagged on unwind.
          case 'G':
            break;
          case 'C':
            CompactUnwind = true;
            break;
          }
        }

        if (AugmentationLength) {
          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
          AugmentationData = Data.getData().slice(StartAugmentationOffset,
                                                  EndAugmentationOffset);
        }
      }

      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, Length, Version, AugmentationString,
          AddressSize, SegmentDescriptorSize, CodeAlignmentFactor,
          DataAlignmentFactor, ReturnAddressRegister, AugmentationData,
          FDEPointerEncoding, LSDAPointerEncoding, Personality,
          PersonalityEncoding, CompactUnwind, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.emplace_back(std::move(Cie));

      // There's no CFI program in the CIE for compact unwind descriptors.
      if (CompactUnwind)
        Offset = EndStructureOffset;
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = 0;
      uint64_t AddressRange = 0;
      SmallVector<FDECompactUnwind, 2> CompactUnwindData;
      std::optional<uint64_t> LSDAAddress;
      CIE *Cie = CIEs[IsEH ? (StartStructureOffset - CIEPointer) : CIEPointer];

      if (IsEH) {
        // The address size is encoded in the CIE we reference.
        if (!Cie)
          return createStringError(errc::invalid_argument,
                                   "parsing FDE data at 0x%" PRIx64
                                   " failed due to missing CIE",
                                   StartOffset);
        if (auto Val =
                Data.getEncodedPointer(&Offset, Cie->getFDEPointerEncoding(),
                                       EHFrameAddress + Offset)) {
          InitialLocation = *Val;
        }
        if (auto Val = Data.getEncodedPointer(
                &Offset, Cie->getFDEPointerEncoding(), 0)) {
          AddressRange = *Val;
        }

        StringRef AugmentationString = Cie->getAugmentationString();
        if (!AugmentationString.empty()) {
          // Parse the augmentation length and data for this FDE.
          uint64_t AugmentationLength = Data.getULEB128(&Offset);

          uint64_t EndAugmentationOffset = Offset + AugmentationLength;

          // Decode the LSDA if the CIE augmentation string said we should.
          if (Cie->getLSDAPointerEncoding() != DW_EH_PE_omit) {
            LSDAAddress = Data.getEncodedPointer(
                &Offset, Cie->getLSDAPointerEncoding(),
                EHFrameAddress ? Offset + EHFrameAddress : 0);
          }

          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
        }
        CompactUnwind = Cie->getCompactUnwind();
        // dbgs() << Offset << " " << EndStructureOffset << " A\n";
        if (CompactUnwind) {
          // dbgs() << ""
          // dbgs() << Offset << " " << EndStructureOffset << " X\n";
          // uint64_t OffCopy = Offset;
          // StringRef Bytes = Data.getBytes(&OffCopy, EndStructureOffset - Offset);
          // dbgs() << format_bytes({(uint8_t*)Bytes.data(), Bytes.size()}) << "\n\n";
          while (Offset < EndStructureOffset) {
            uint64_t Skip = Data.getULEB128(&Offset);
            // dbgs() << Offset << " " << EndStructureOffset << " Y\n";
            // A zero skip would indicate that the previous descriptor has no
            // size; hence take this as end of the descriptor sequence.
            // XXX: MC sometimes emits zero skips
            if (!CompactUnwindData.empty() && Skip == 0 && Offset + 8 > EndStructureOffset) {
              Offset = EndStructureOffset;
              break;
            }
            // Skip *= Cie->getCodeAlignmentFactor();

            uint64_t Desc = 0; // Default to NULL descriptor.
            if (Offset + sizeof(uint64_t) <= EndStructureOffset)
              Desc = Data.getU64(&Offset);
            else
              Offset = EndStructureOffset;

            // dbgs() << Offset << " " << EndStructureOffset << " Z\n";
            CompactUnwindData.emplace_back(Skip, Desc);
          }
          // dbgs() << Offset << " " << EndStructureOffset << " W\n";
        }
      } else {
        InitialLocation = Data.getRelocatedAddress(&Offset);
        AddressRange = Data.getRelocatedAddress(&Offset);
      }

      Entries.emplace_back(new FDE(IsDWARF64, StartOffset, Length, CIEPointer,
                                   InitialLocation, AddressRange,
                                   std::move(CompactUnwindData), Cie, LSDAAddress,
                                   Arch));
    }

    if (!CompactUnwind) {
      if (Error E =
              Entries.back()->cfis().parse(Data, &Offset, EndStructureOffset))
        return E;
    }
    // dbgs() << int(CompactUnwind) << " " << Offset << " " << EndStructureOffset << " XXXX\n";

    if (Offset != EndStructureOffset)
      return createStringError(
          errc::invalid_argument,
          "parsing entry instructions at 0x%" PRIx64 " failed", StartOffset);
  }

  return Error::success();
}

FrameEntry *DWARFDebugFrame::getEntryAtOffset(uint64_t Offset) const {
  auto It = partition_point(Entries, [=](const std::unique_ptr<FrameEntry> &E) {
    return E->getOffset() < Offset;
  });
  if (It != Entries.end() && (*It)->getOffset() == Offset)
    return It->get();
  return nullptr;
}

void DWARFDebugFrame::dump(raw_ostream &OS, DIDumpOptions DumpOpts,
                           std::optional<uint64_t> Offset) const {
  DumpOpts.IsEH = IsEH;
  if (Offset) {
    if (auto *Entry = getEntryAtOffset(*Offset))
      Entry->dump(OS, DumpOpts);
    return;
  }

  OS << "\n";
  for (const auto &Entry : Entries)
    Entry->dump(OS, DumpOpts);
}
