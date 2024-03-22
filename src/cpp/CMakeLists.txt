cmake_minimum_required(VERSION 3.0)
project(LLVMBasicBlockCounter)

find_package(LLVM REQUIRED CONFIG)
set(CMAKE_CXX_COMPILER "clang++")  # Set the C++ compiler to clang++
set(CMAKE_CXX_STANDARD 14)

set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})



add_executable(count_llvm_ir_bb count_llvm_ir_bb.cpp)

llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(count_llvm_ir_bb ${llvm_libs} ${CLANG_LIBS})



target_link_libraries(count_llvm_ir_bb
  LLVMipo
  LLVMScalarOpts
  LLVMInstCombine
  LLVMTransformUtils
  LLVMAnalysis
  LLVMTarget
  LLVMOption # Support
  LLVMMCParser # MC, Support
  LLVMMC # Object, Support
  LLVMObject # BitReader, Core, Support
  LLVMBitReader # Core, Support
  LLVMCore # Support
  LLVMSupport
)
