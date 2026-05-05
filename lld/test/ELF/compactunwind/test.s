# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple x86_64 --elf-compact-unwind < %t/f1.s -o %t/f1.o
# RUN: llvm-readelf --unwind %t/f1.o | FileCheck %s --check-prefix=F1
# F1:       [0x14] FDE
# F1-NEXT:    initial_location: 0x0
# F1-NEXT:    address_range: 0x6 (end : 0x6)
# F1-NEXT:    unwind_descriptor: +0x0 0000010420000001
# F1-EMPTY:

# RUN: llvm-mc -filetype=obj -triple x86_64 --elf-compact-unwind < %t/f2.s -o %t/f2.o
# RUN: llvm-readelf --unwind %t/f2.o | FileCheck %s --check-prefix=F2
# F2:       FDE
# F2-NEXT:    initial_location: 0x0
# F2-NEXT:    address_range: 0x8 (end : 0x8)
# F2-NEXT:    unwind_descriptor: +0x0 0000010420000001
# F2-EMPTY:
# F2:       FDE
# F2-NEXT:    initial_location: 0x10
# F2-NEXT:    address_range: 0xa (end : 0x1a)
# F2-NEXT:    unwind_descriptor: +0x0 0000010420000001
# F2-NEXT:    unwind_descriptor: +0x8 0000000020000001
# F2-EMPTY:
# F2:       FDE
# F2-NEXT:    initial_location: 0x20
# F2-NEXT:    address_range: 0x8 (end : 0x28)
# F2-NEXT:    unwind_descriptor: +0x0 0000010420000001
# F2-EMPTY:

# RUN: ld.lld --eh-frame-hdr %t/f1.o %t/f2.o -o %t/exe
# RUN: llvm-readelf --unwind %t/exe | FileCheck %s --check-prefix=EXE
# EXE:      EHFrameHeader {
# EXE:        Header {
# EXE-NEXT:     version: 2
# EXE-NEXT:     ptr_enc: 0x1b
# EXE-NEXT:     page 0 {
# EXE-NEXT:       pc: 0x[[#%x,ADDR:]]
# EXE-NEXT:       entry: 0x[[#ADDR]] 00000b0420000001
# EXE-NEXT:       entry: 0x[[#ADDR+0x10]] 0000090420000001
# EXE-NEXT:       entry: 0x[[#ADDR+0x20]] 0000010420000001
# EXE-NEXT:       entry: 0x[[#ADDR+0x28]] 0000000020000001
# EXE-NEXT:       entry: 0x[[#ADDR+0x30]] 0000010420000001
# EXE-NEXT:       entry: 0x[[#ADDR+0x38]] 0000000000000001
# EXE-NEXT:     }
# EXE-NEXT:     page 1 {
# EXE-NEXT:       pc: 0x[[#ADDR+0x38]]
# EXE-NEXT:     }
# EXE-NEXT:   }
# EXE-NEXT: }

#--- f1.s
    .globl f1
    .p2align 4
    .type f1,@function
f1:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
    .size f1, .-f1
    .cfi_endproc

#--- f2.s
    .globl f2a
    .p2align 4
    .type f2,@function
f2a:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    xor %eax, %eax
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
    .size f2a, .-f2a
    .cfi_endproc

    .globl f2b
    .p2align 4
    .type f2,@function
f2b:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    jmp 2f
1:
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
2:
    .cfi_def_cfa_offset 16
    .cfi_def_cfa_register %rbp
    .cfi_offset %rbp, -16
    jmp 1b
    .size f2b, .-f2b
    .cfi_endproc

    .globl f2c
    .p2align 4
    .type f2,@function
f2c:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    movq    %rsp, %rbp
    .cfi_def_cfa_register %rbp
    xor %eax, %eax
    popq    %rbp
    .cfi_def_cfa %rsp, 8
    retq
    .size f2c, .-f2c
    .cfi_endproc
