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


using namespace llvm;

static std::unique_ptr<Module> readModule(LLVMContext &Context, StringRef Name) {
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseIRFile(Name, Err, Context);
    if (!M) {
        Err.print("count_llvm_ir_bb", errs());
    }
    return M;
}


cl::OptionCategory CountBBCategory("count_llvm_ir_bb options");
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(CountBBCategory));

int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(CountBBCategory);
    cl::ParseCommandLineOptions(argc, argv, "count_llvm_ir_bb\n");

    LLVMContext Context;
    std::unique_ptr<Module> M = readModule(Context, InputFilename);
    if (!M) {
        return 1;
    }

    int bb_count = 0;
    for (auto &F : *M) {
        bb_count += F.size();
    }

    // WithColor::note() << "bbcount: " << bb_count << "\n";
    printf("%d\n", bb_count);
    return 0;
}
