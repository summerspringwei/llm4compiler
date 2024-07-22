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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <iostream>
#include <sstream>
#include <set>

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

bool isLlvmDbgCallInst(Instruction& inst){
    if (auto *CI = dyn_cast<llvm::CallInst>(&inst)){
        if (CI->getCalledFunction()->getName() == "llvm.dbg.value"){
            return true;
        }
    }
    return false;
}


unsigned getInstLoc(Instruction& inst){
    if (auto Loc = inst.getDebugLoc()) {
        return Loc.getLine();
    }
    return 0;
}

const DISubprogram *getDISubprogram(const DIScope *Scope) {
    if (!Scope) return nullptr;

    if (auto *Subprogram = dyn_cast<DISubprogram>(Scope)){
        errs() << "Subprogram: " << Subprogram->getName() << "\n";
        return Subprogram;
    }
        
    if (auto *Block = dyn_cast<DILexicalBlock>(Scope)){
        errs() << "Block: " << Block << "\n";
        return getDISubprogram(Block->getScope());
    }
        
    if (auto *BlockFile = dyn_cast<DILexicalBlockFile>(Scope)){
        errs() << "BlockFile: " << BlockFile << "\n";
        return getDISubprogram(BlockFile->getScope());
    }

    return nullptr;
}

const DILexicalBlock *getTopLevelLexicalBlock(const DIScope *Scope, const DISubprogram *subprogram) {
    if (!Scope) return nullptr;

    if (auto *Block = dyn_cast<DILexicalBlock>(Scope)){
        if (auto *subprogram_of_block = dyn_cast<DISubprogram>(Block->getScope())){
            if(subprogram_of_block == subprogram){
                return Block;
            }else{
                throw std::runtime_error("The DILexicalBlock is not in the same subprogram");
            }
        }
        return getTopLevelLexicalBlock(Block->getScope(), subprogram);
    }
    
    return nullptr;
}

void getDbgOfBBs(std::unique_ptr<Module> M){
    const llvm::DISubprogram* subprogram;
    bool findSubprogram = false;
    // 1. Get the DISubprogram of the function
    for (auto &F : *M) {
        for(auto &BB : F){
            for(auto &I : BB){
                if (DILocation *Loc = I.getDebugLoc()){
                    subprogram = getDISubprogram(Loc->getScope());
                    if (subprogram) {
                        errs() << "Subprogram: " << subprogram->getName() << "\n";
                        findSubprogram = true;
                        break;
                    }
                }
            }
            if(findSubprogram){
                break;
            }
        }
    }
    // 2. Get the top-level DILexicalBlock of the function    
    std::set<const DILexicalBlock*> topLevelBlocks;
    for (auto &F : *M) {
        for(auto &BB : F){
            for(auto &I : BB){
                if (DILocation *Loc = I.getDebugLoc()){
                    auto* result = getTopLevelLexicalBlock(Loc->getScope(), subprogram);
                    if(result){
                        topLevelBlocks.insert(result);
                    }
                }
            }
        }
    }
    errs() << "num of toplevel bb:" << topLevelBlocks.size() << "\n";
    for(auto* b: topLevelBlocks){
        errs() << b->getLine() << "\n";
    }
    // 3. Check whether all the instruction of a BasicBlock belongs to the same top-level DILexicalBlock
}


int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(ExtractIRDbg);
    cl::ParseCommandLineOptions(argc, argv, "extract_llvm_ir_dbg\n");

    LLVMContext Context;
    std::unique_ptr<Module> M = readModule(Context, InputFilename);
    if (!M) {
        return 1;
    }
    // We need to first strip the `call void @llvm.dbg.value`
    // for (auto &F : *M) {
    //     for(auto &BB : F){
    //         // Firstly, we try to get the first and last instruction's debug info in a basic block
    //         if (BB.empty()){
    //             continue;
    //         }
    //         if (BB.size() == 1){
    //             llvm::Instruction* I = &(BB.front());
    //             auto line = getInstLoc(*I);
    //             if (line) {
    //                 errs() << "BB with one Instruction: " << *I << "\t" << line << "\n";
    //             }
    //         }else{
    //             llvm::Instruction* start_inst_ptr=NULL;
    //             llvm::Instruction* end_inst_ptr=NULL;
    //             for (auto& inst : BB){
    //                 if (! isLlvmDbgCallInst(inst) && start_inst_ptr == NULL){
    //                     start_inst_ptr = &inst;
    //                     break;
    //                 }
    //             }
    //             errs() << "BB with more Instruction: " << *start_inst_ptr << "\t" << getInstLoc(*start_inst_ptr) << "\n";
    //             // auto& end = BB.
    //             if (DILocation *Loc = (*start_inst_ptr).getDebugLoc()) {
    //                 DIScope *Scope = Loc->getScope();
    //                 getDISubprogram(Scope);
    //                 // if (DILexicalBlock *LB = dyn_cast<DILexicalBlock>(Scope)) {
    //                 //     getRootScope(LB);
    //                 // }
    //             }
    //         }
    //     }
    // }
    getDbgOfBBs(std::move(M));
    return 0;
}
