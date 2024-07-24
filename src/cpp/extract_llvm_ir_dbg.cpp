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
#include <unordered_map>

using namespace llvm;


class IntegerRange {
private:
    int start;
    int end;

public:
    // Constructor
    IntegerRange(unsigned start, unsigned end) {
        if (start > end) {
            throw std::invalid_argument("Start should not be greater than end.");
        }
        this->start = start;
        this->end = end;
    }

    IntegerRange() : start(102400), end(0) {}

    IntegerRange(const IntegerRange &other) : start(other.start), end(other.end) {}

    IntegerRange& operator=(const IntegerRange& other) {
        if (this != &other) {
            this->start = other.start;
            this->end = other.end;
        }
        return *this;
    }

    // Getters
    unsigned getStart() const {
        return start;
    }

    unsigned getEnd() const {
        return end;
    }

    bool insert(unsigned value) {
        if (value < start) {
            start = value;
            return true;
        } else if (value > end) {
            end = value;
            return true;
        }
        return false;
    }

    // Member function to check if a number is within the range
    bool contains(unsigned value) const {
        return value >= start && value <= end;
    }

    // Member function to print the range
    std::string rangeStr() {
        std::stringstream ss;
        ss << "[" << start << ", " << end << "]";
        return ss.str();
    }

};

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

void getDIOfBBs(std::unique_ptr<Module> M){
    const llvm::DISubprogram* subprogram = nullptr;
    // 1. Get the DISubprogram of the function
    for (auto &F : *M) { // Note, here we assume there is only one function in the module
        bool findSubprogram = false;
        for(auto &BB : F){
            for(auto &I : BB){
                if (DILocation *Loc = I.getDebugLoc()){
                    if ((subprogram = getDISubprogram(Loc->getScope()))) {
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
    if (!subprogram){
        throw std::runtime_error("Cannot find the DISubprogram of the function");
    }
    // 2. Get the top-level DILexicalBlock of the function    
    // Record each basic blocks's top-level DILexicalBlock
    // if a BB has multiple top-level DILexicalBlock, we need to report an error
    // otherwise, what we should do is to record the top-level DILexicalBlock of each BB
    // We also extract the range of DILexicalBlock in the source file  
    std::unordered_map<const BasicBlock*, const DILexicalBlock*> bbToLexicalBlock;
    std::unordered_map<const DILexicalBlock*, IntegerRange> lexicalBlockToRange;

    std::set<const DILexicalBlock*> topLevelBlocks;
    for (auto &F : *M) {
        for(auto &BB : F){
            for(auto &I : BB){
                DILocation *Loc = nullptr;
                if ((Loc = I.getDebugLoc()) && !isLlvmDbgCallInst(I)){
                    // const DILexicalBlock * lexical_block = nullptr;
                    const DILexicalBlock * top_lexical_block = nullptr;
                    if((top_lexical_block = getTopLevelLexicalBlock(Loc->getScope(), subprogram))){
                        //TODO(Chunwei) Compute the range here
                        if(lexicalBlockToRange.find(top_lexical_block) == lexicalBlockToRange.end()){
                            // lexicalBlockToRange[top_lexical_block] = IntegerRange(Loc->getLine(), Loc->getLine());
                            errs() << "Create range" << Loc->getLine() << "\n";
                            lexicalBlockToRange.insert({top_lexical_block, IntegerRange(Loc->getLine(), Loc->getLine())});
                        }else{
                            lexicalBlockToRange[top_lexical_block].insert(Loc->getLine());
                            errs() << "Update range" << Loc->getLine() << "\n";
                        }
                        // topLevelBlocks.insert(top_lexical_block);
                        if(bbToLexicalBlock.find(&BB) != bbToLexicalBlock.end() && bbToLexicalBlock[&BB] != top_lexical_block){
                            throw std::runtime_error("The BB has multiple top-level DILexicalBlock");
                        }else{
                            bbToLexicalBlock[&BB] = top_lexical_block;
                        }
                    }
                }
            }
        }
    }

    errs() << "num of toplevel bb:" << lexicalBlockToRange.size() << "\n";
    for(auto b: lexicalBlockToRange){
        errs() << b.first->getLine() << b.second.rangeStr() << "\n";
    }
    for (auto &F : *M) {
        for(auto &BB : F){
            if (bbToLexicalBlock.find(&BB) == bbToLexicalBlock.end()){
                errs() << "BB without top-level DILexicalBlock: " << BB << "\n";
            }else{
                errs() << BB << "\n ^^^" << bbToLexicalBlock[&BB]->getLine() << "to" << "bbToLexicalBlock[&BB]->getScopeLine()" << "^^^\n";
            }
        }
    }
}


int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(ExtractIRDbg);
    cl::ParseCommandLineOptions(argc, argv, "extract_llvm_ir_dbg\n");

    LLVMContext Context;
    std::unique_ptr<Module> M = readModule(Context, InputFilename);
    if (!M) {
        return 1;
    }
    getDIOfBBs(std::move(M));
    return 0;
}
