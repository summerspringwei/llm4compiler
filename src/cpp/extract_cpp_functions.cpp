#include <iostream>
#include <fstream>
#include <string>
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/CommonOptionsParser.h"

using namespace clang;
using namespace clang::tooling;

// AST visitor to extract function declarations
class FunctionExtractor : public RecursiveASTVisitor<FunctionExtractor> {
public:
    explicit FunctionExtractor(ASTContext *Context, std::ofstream &outFile)
        : Context(Context), outFile(outFile) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        // Ignore function declarations without a body
        if (FD->hasBody()) {
            // Write function signature to output file
            outFile << FD->getReturnType().getAsString() << " " << FD->getNameAsString() << "(";
            for (auto param : FD->parameters()) {
                outFile << param->getType().getAsString() << " " << param->getNameAsString();
                if (param != *(FD->param_end()) - 1)
                    outFile << ", ";
            }
            outFile << ");\n";
        }
        return true;
    }

private:
    ASTContext *Context;
    std::ofstream &outFile;
};

static llvm::cl::OptionCategory MyOpts("Ignored");

// Main function to extract functions and their declarations from a C++ file
int main(int argc, const char **argv) {
    if (argc < 2) {
        llvm::errs() << "Usage: " << argv[0] << " <input file>\n";
        return 1;
    }

    std::string InputFilename = argv[1];

    // Create Clang tool
    ClangTool Tool();

    // Create AST unit from the input file
    CommonOptionsParser opt_prs(argc, argv, MyOpts);
    ClangTool tool(opt_prs.getCompilations(), opt_prs.getSourcePathList());
    using ast_vec_t = std::vector<std::unique_ptr<ASTUnit>>;
    ast_vec_t asts;
    tool.buildASTs(asts); 
    // std::unique_ptr<clang::ASTUnit> AST(clang::tooling::buildASTFromCode(code));
    auto AST=asts[0].get();
    // Open output file
    std::ofstream outFile("output.cpp");
    if (!outFile.is_open()) {
        llvm::errs() << "Unable to open output file\n";
        return 1;
    }

    // Create AST visitor and traverse the AST
    FunctionExtractor extractor(&AST->getASTContext(), outFile);
    extractor.TraverseDecl(AST->getASTContext().getTranslationUnitDecl());

    // Close output file
    outFile.close();

    return 0;
}
