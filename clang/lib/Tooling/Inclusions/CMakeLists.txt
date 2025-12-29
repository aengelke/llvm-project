set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangToolingInclusions
  HeaderAnalysis.cpp
  HeaderIncludes.cpp
  IncludeStyle.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangToolingCore
  )

  add_subdirectory(Stdlib)
