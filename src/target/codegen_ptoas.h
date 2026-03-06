// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ptoas.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_PTOAS_H_
#define TVM_TL_TARGET_CODEGEN_PTOAS_H_

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <PTO/IR/PTO.h>

#include <tvm/target/codegen.h>
#include <tvm/tir/function.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>

#include "codegen_ascend_pto.h"

namespace tvm {
namespace codegen {

using namespace tir;

class CodeGenPTOAS : public tvm::tir::ExprFunctor<mlir::Value(const tvm::PrimExpr&)>,
                     public tvm::tir::StmtFunctor<void(const tvm::tir::Stmt&)> {
public:
  CodeGenPTOAS();
  virtual ~CodeGenPTOAS();
  virtual void Init();
  virtual std::string Finish();
  std::string GetHostFn() const { return hostfn; }

  // Resolves MLIR type from TVM Type (which can be PointerType)
  mlir::Type resolveType(const tvm::Type &type);
  // Resolves MLIR type from primitive DataType
  mlir::Type resolvePrimitiveType(const tvm::runtime::DataType &dtype);

  // Resolve an MLIR type that is legal for the arith dialect given a TVM DataType.
  // (Integers are returned as signless arith-friendly IntegerType of same width;
  // floats/index are returned unchanged.)
  mlir::Type resolveArithType(const tvm::runtime::DataType &dtype);

  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

public:
  // TVM Visitors
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const SeqStmtNode *op) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;

  mlir::Value VisitExpr_(const BufferLoadNode *op) final;
  mlir::Value VisitExpr_(const CallNode *op) final;
  mlir::Value VisitExpr_(const VarNode *op) final;
  mlir::Value VisitExpr_(const IntImmNode *op) final;
  mlir::Value VisitExpr_(const CastNode *op) final;
  mlir::Value VisitExpr_(const FloatImmNode *op) final;
  mlir::Value VisitExpr_(const StringImmNode *op) final;
  mlir::Value VisitExpr_(const AddNode *op) final;
  mlir::Value VisitExpr_(const SubNode *op) final;
  mlir::Value VisitExpr_(const MulNode *op) final;
  mlir::Value VisitExpr_(const DivNode *op) final;
  mlir::Value VisitExpr_(const ModNode *op) final;
  mlir::Value VisitExpr_(const MinNode *op) final;
  mlir::Value VisitExpr_(const MaxNode *op) final;
  mlir::Value VisitExpr_(const LTNode *op) final;
  mlir::Value VisitExpr_(const LENode *op) final;
  mlir::Value VisitExpr_(const GTNode *op) final;
  mlir::Value VisitExpr_(const GENode *op) final;
  mlir::Value VisitExpr_(const EQNode *op) final;
  mlir::Value VisitExpr_(const NENode *op) final;
  mlir::Value VisitExpr_(const AndNode *op) final;
  mlir::Value VisitExpr_(const OrNode *op) final;
  mlir::Value VisitExpr_(const NotNode *op) final;
  mlir::Value VisitExpr_(const SelectNode *op) final;
  mlir::Value VisitExpr_(const LetNode *op) final;
  mlir::Value VisitExpr_(const FloorDivNode *op) final;
  mlir::Value VisitExpr_(const FloorModNode *op) final;

public:
  struct cgsymbol {
    bool need_reshape_for_reduce = false;
    mlir::Value sym;
  };
  // MLIR specific helpers
  mlir::Value CallExternCodegen(const CallNode *op);
  mlir::Value GetAsTile(const PrimExpr &op);
  const tvm::tir::VarNode* GetBufferVar(const PrimExpr &op);

  void UbShapeInputCheck(const AllocateNode *op);
  bool ValidLayoutEnabled(const AllocateNode *op);

  struct resolvedBuffer {
    std::vector<PrimExpr> shape;
  };

  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unordered_map<const tvm::tir::VarNode*, cgsymbol> symbolTable;
  std::unordered_map<const tvm::tir::VarNode*, resolvedBuffer> bufferTable;
  std::string source_scope;

private:
  Map<Var, PrimExpr> address_map_;
  Map<Var, Array<PrimExpr>> buffer_shapes_;
  std::string core_num_;
  CodeGenTileLangAscendPto CGC;
  std::string hostfn;
};

class CodeGenTileLangPTOAS final : public CodeGenPTOAS {
public:
  using super = CodeGenPTOAS;
  CodeGenTileLangPTOAS(std::string platform);
  std::string Finish();
};

} // namespace codegen
} // namespace tvm
#endif // TVM_TL_TARGET_CODEGEN_ASCEND_PTO_H_
