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
    
    
    std::stringstream ss;
    for (auto &F : *M) {
        int bb_count = 0;
        bb_count += F.size();
        if (bb_count > 0){
            size_t *bb_size_list = new size_t[F.size()];
            int idx = 0;
            for (auto &BB : F) {
                bb_size_list[idx++] = BB.size();
            }
            
            ss << "{" << "\"func_name\"" << ": \"" << F.getName().str() << "\" ,";
            ss << "\"bbcount\"" << ":" << bb_count << "," << "\"bb_list_size\"" << ": [";
            for(int i=0; i<F.size(); i++) {
                ss << bb_size_list[i];
                if(i != F.size()-1) {
                    ss << ",";
                }
            }ss << "]}\n";
        }
    }
        
    // WithColor::note() << "bbcount: " << bb_count << "\n";
    printf("%s\n", ss.str().c_str());
    return 0;
}
