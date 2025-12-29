set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingCore
  Diagnostic.cpp
  Replacement.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  )
