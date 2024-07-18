#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/WithColor.h"
#include <iostream>
#include <sstream>

using namespace llvm;

static std::unique_ptr<Module> readModule(LLVMContext &Context, StringRef Name) {
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseIRFile(Name, Err, Context);
    if (!M) {
        Err.print("extract_llvm_ir_dbg", errs());
    }
    return M;
}


cl::OptionCategory ExtractIRDbg("extract_llvm_ir_dbg options");
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(ExtractIRDbg));

int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(ExtractIRDbg);
    cl::ParseCommandLineOptions(argc, argv, "extract_llvm_ir_dbg\n");

    LLVMContext Context;
    std::unique_ptr<Module> M = readModule(Context, InputFilename);
    if (!M) {
        return 1;
    }

    std::stringstream ss;
    for (auto &F : *M) {
        for(auto &BB : F){
        for (auto &I : BB) {
            // Check if the instruction has debug information
            if (auto Loc = I.getDebugLoc()) {
                unsigned Line = Loc.getLine();
                // unsigned Col = Loc.getColumn();
                // StringRef File = Loc->getFilename();
                // StringRef Dir = Loc->getDirectory();
                
                // Print the debug information
                errs() << "Instruction: " << I << "\n";
                errs() << "Debug Info - Line: " << Line 
                        // << ", Column: " << Col 
                        // << ", File: " << File 
                        // << ", Directory: " << Dir
                        << "\n";
            }
        }
        }
    }
    return 0;
}
