#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>
#include <vector>
#include <sstream>

using namespace llvm;

static std::unique_ptr<Module> readModule(LLVMContext &Context, StringRef Name) {
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseIRFile(Name, Err, Context);
    if (!M) {
        Err.print("count_llvm_ir_bb", errs());
    }
    return M;
}


int removeDeclaredOnlyFunctions(Module *M) {
    std::vector<Function *> toRemove;

    // Collect declared-only functions
    for (auto &F : *M) {
        if (F.isDeclaration()) {
            toRemove.push_back(&F);
        }
    }

    // Remove collected functions
    for (Function *F : toRemove) {
        F->eraseFromParent();
    }
    return toRemove.size();
}

cl::OptionCategory CountBBCategory("filter_declare_only_functions options");
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(CountBBCategory));
static cl::opt<std::string> OutputFilename(cl::Positional, cl::desc("<output file>"), cl::Required, cl::cat(CountBBCategory));

int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(CountBBCategory);
    cl::ParseCommandLineOptions(argc, argv, "filter_declare_only_functions\n");

    LLVMContext Context;
    std::unique_ptr<Module> M = readModule(Context, InputFilename);
    if (!M) {
        return 1;
    }
    
    auto count = removeDeclaredOnlyFunctions(M.get());
    printf("Removed %d declare only functions\n", count);

    // Save the module to a human-readable file
    std::error_code EC;
    llvm::raw_fd_ostream OS(OutputFilename, EC, llvm::sys::fs::OF_None);
    if (!EC) {
        M->print(OS, nullptr);
    } else {
        llvm::errs() << "Error: " << EC.message() << "\n";
        return 1;
    }
    return 0;
}
