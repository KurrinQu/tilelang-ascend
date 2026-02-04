// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_ptoas.cc
 */

#include "codegen_ptoas.h"
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <PTO/IR/PTO.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>

#include "../op/ascend.h"
#include "../op/builtin.h"

#include "arith/pattern_match.h"

#define DEC_STR_TO_HEX_STR(dec_str) \
  ([](const std::string& s){ std::stringstream ss; \
  ss << std::showbase << std::hex << std::uppercase << std::stoi(s); \
  return ss.str(); }(dec_str))

namespace tvm {
namespace codegen {

CodeGenPTOAS::CodeGenPTOAS() : builder(&context) {
  context.loadDialect<mlir::func::FuncDialect,
                      mlir::arith::ArithDialect,
                      mlir::pto::PTODialect,
                      mlir::scf::SCFDialect>();
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
};
CodeGenPTOAS::~CodeGenPTOAS() = default;
void CodeGenPTOAS::Init() {}

std::string CodeGenPTOAS::Finish() {
  std::string code;
  llvm::raw_string_ostream os(code);
  if (mlir::failed(mlir::verify(*module))) {
    LOG(ERROR) << "Module verification failed.";
  }
  if (module) {
    module->print(os);
  }
  return code;
}

mlir::Type CodeGenPTOAS::resolveType(const tvm::Type &type) {
  // Check if it's a PointerType
  if (const auto * ptrNode = type.as<tvm::PointerTypeNode>()) {
    mlir::Type elementType = resolveType(ptrNode->element_type);
    return mlir::pto::PtrType::get(&context, elementType);
  }

  // If it's a PrimType, extract the DataType and resolve it
  if (const auto *primNode = type.as<tvm::PrimTypeNode>()) {
    return resolveArithType(primNode->dtype);
  }

  LOG(FATAL) << "Unsupported type: " << type;
}

mlir::Type CodeGenPTOAS::resolvePrimitiveType(const tvm::runtime::DataType &dtype) {
  // Handle Integer/UInt
  if (dtype.is_int() || dtype.is_uint()) {
    return builder.getIntegerType(dtype.bits(), dtype.is_int());
  } else if (dtype.is_float16()) {
    return builder.getF16Type();
  } else if (dtype.is_bfloat16()) {
    return builder.getBF16Type();
  } else if (dtype.is_float()) {
    return builder.getF32Type();
  } else {
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return nullptr;
  }
}

static std::tuple<int, int, int, bool> ExtractTemplateParamsForSliceBuffer(const std::string& op_name) {
    int second_param = 0;
    int third_param = 0;
    int forth_param = 0;
    size_t left = op_name.find('<');
    size_t right = op_name.find('>');

    if (left == std::string::npos || right == std::string::npos || left >= right) {
        return std::make_tuple(second_param, third_param, forth_param, false);
    }

    std::string params_str = op_name.substr(left + 1, right - left - 1);
    std::vector<std::string> params;
    size_t start = 0;
    size_t comma = 0;
    while ((comma = params_str.find(',', start)) != std::string::npos) {
        std::string param = params_str.substr(start, comma - start);
        param.erase(0, param.find_first_not_of(" \t"));
        param.erase(param.find_last_not_of(" \t") + 1);
        params.push_back(param);
        start = comma + 1;
    }

    std::string last_param = params_str.substr(start);
    last_param.erase(0, last_param.find_first_not_of(" \t"));
    last_param.erase(last_param.find_last_not_of(" \t") + 1);
    params.push_back(last_param);

    if (params.size() >= 4) {
        try {
            second_param = std::stoi(params[1]);
            third_param = std::stoi(params[2]);
            forth_param = std::stoi(params[3]);
            return std::make_tuple(second_param, third_param, forth_param, true);
        } catch (const std::exception& e) {
            return std::make_tuple(second_param, third_param, forth_param, false);
        }
    } else {
      ICHECK(false) << "reduce params less than 4.";
    }
    return std::make_tuple(second_param, third_param, forth_param, false);
}

// Normalize a given MLIR type to a form suitable for arith dialect ops.
// Input: any mlir::Type
// Output: canonical "arith-friendly" mlir::Type (e.g. signless integer of same width,
//         floats/index returned as-is). Other types are returned unchanged.
static mlir::Type NormalizeArithType(mlir::Type inType, mlir::OpBuilder &builder) {
  // Integers -> signless integer of same width (arith expects signless ints)
  if (mlir::isa<mlir::IntegerType>(inType)) {
    unsigned width = mlir::cast<mlir::IntegerType>(inType).getWidth();
    return mlir::IntegerType::get(builder.getContext(), width);
  }
  // Floats are already fine for arith ops
  if (mlir::isa<mlir::FloatType>(inType)) {
    return inType;
  }
  // Index type is a distinct legal type for many arith ops
  if (inType.isIndex()) {
    return inType;
  }
  // Fallback: return the input type unchanged for cases we don't explicitly canonicalize.
  return inType;
}

// Convert a TVM runtime::DataType into an MLIR type that is legal for arith ops.
// This is a convenience wrapper: it first resolves the primitive MLIR type
// (resolvePrimitiveType) and then canonicalizes it for arith (signless ints, etc).
mlir::Type CodeGenPTOAS::resolveArithType(const tvm::runtime::DataType &dtype) {
  // Resolve the primitive MLIR type first (may be signed integer, float, etc).
  mlir::Type prim = resolvePrimitiveType(dtype);
  // Normalize to arith-friendly form (signless integers, float/index pass-through).
  return NormalizeArithType(prim, builder);
}

// Helper: convert `val` to `dstType`. `dstSigned` indicates desired signedness for integer dst.
static mlir::Value CastVal(mlir::OpBuilder &builder,
                           mlir::Location loc,
                           mlir::Value val,
                           mlir::Type dstType,
                           bool dstSigned) {
  if (!val) return mlir::Value();
  mlir::Type srcType = val.getType();
  if (srcType == dstType) return val;

  // index <-> integer conversions
  if (srcType.isIndex() && mlir::isa<mlir::IntegerType>(dstType)) {
    return builder.create<mlir::arith::IndexCastOp>(loc, dstType, val).getResult();
  }
  if (mlir::isa<mlir::IntegerType>(srcType) && dstType.isIndex()) {
    // when converting to index, use dstType (index) as-is
    return builder.create<mlir::arith::IndexCastOp>(loc, dstType, val).getResult();
  }

  // integer -> integer (extend / trunc)
  if (mlir::isa<mlir::IntegerType>(srcType) && mlir::isa<mlir::IntegerType>(dstType)) {
    int srcBits = mlir::cast<mlir::IntegerType>(srcType).getWidth();
    int dstBits = mlir::cast<mlir::IntegerType>(dstType).getWidth();
    if (dstBits > srcBits) {
      // choose signed/unsigned extend based on requested signedness
      return dstSigned
          ? builder.create<mlir::arith::ExtSIOp>(loc, dstType, val).getResult()
          : builder.create<mlir::arith::ExtUIOp>(loc, dstType, val).getResult();
    } else if (dstBits < srcBits) {
      return builder.create<mlir::arith::TruncIOp>(loc, dstType, val).getResult();
    } else {
      // same width but ensure canonical unsigned destination type
      return builder.create<mlir::arith::IndexCastOp>(loc, dstType, val).getResult();
    }
  }

  // float <-> integer conversions
  if (mlir::isa<mlir::FloatType>(srcType) && mlir::isa<mlir::IntegerType>(dstType)) {
    // FP -> int: pick signed/unsigned op, but destination type must be unsigned canonical
    return dstSigned
        ? builder.create<mlir::arith::FPToSIOp>(loc, dstType, val).getResult()
        : builder.create<mlir::arith::FPToUIOp>(loc, dstType, val).getResult();
  }
  if (mlir::isa<mlir::IntegerType>(srcType) && mlir::isa<mlir::FloatType>(dstType)) {
    // int -> float: choose op based on source signedness (source kept as-is)
    return mlir::cast<mlir::IntegerType>(srcType).isSigned()
        ? builder.create<mlir::arith::SIToFPOp>(loc, dstType, val).getResult()
        : builder.create<mlir::arith::UIToFPOp>(loc, dstType, val).getResult();
  }

  // float -> float (ext / trunc)
  if (mlir::isa<mlir::FloatType>(srcType) && mlir::isa<mlir::FloatType>(dstType)) {
    int srcBits = mlir::cast<mlir::FloatType>(srcType).getWidth();
    int dstBits = mlir::cast<mlir::FloatType>(dstType).getWidth();
    if (dstBits > srcBits) {
      return builder.create<mlir::arith::ExtFOp>(loc, dstType, val).getResult();
    } else {
      return builder.create<mlir::arith::TruncFOp>(loc, dstType, val).getResult();
    }
  }

  // Fallback: if nothing matches, try an IndexCast fallback to normalized dst
  return builder.create<mlir::arith::IndexCastOp>(loc, dstType, val).getResult();
}

template <typename OpTy>
static void EmitUnaryOp(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call) {
  auto res = self->GetAsTile(call->args[0]);
  auto src = self->GetAsTile(call->args[1]);
  builder.create<OpTy>(loc, src, res);
}

template <typename OpTy>
static void EmitUnaryOpWithFAttr(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call, mlir::FloatAttr attr) {
  auto res = self->GetAsTile(call->args[0]);
  auto src = self->GetAsTile(call->args[1]);
  builder.create<OpTy>(loc, src, res, attr);
}

template <typename OpTy>
static void EmitBinaryOpWithFlag(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call, bool flag) {
  auto res = self->GetAsTile(call->args[0]);
  auto lhs = self->GetAsTile(call->args[1]);
  auto rhs = self->GetAsTile(call->args[2]);
  builder.create<OpTy>(loc, lhs, rhs, res, mlir::BoolAttr::get(&self->context, flag));
}

template <typename OpTy>
static void EmitBinaryOp(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call) {
  auto res = self->GetAsTile(call->args[0]);
  auto lhs = self->GetAsTile(call->args[1]);
  auto rhs = self->GetAsTile(call->args[2]);
  builder.create<OpTy>(loc, lhs, rhs, res);
}

template <typename OpTy>
static void EmitScalarBinaryOp(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call) {
  auto res = self->GetAsTile(call->args[0]);
  auto lhs = self->GetAsTile(call->args[1]);
  mlir::Value rhs;
  if (call->args[2].as<CallNode>()) {
    auto dtype = self->resolveArithType(call->args[2].as<CallNode>()->args[0]->dtype);
    auto stile = self->GetAsTile(call->args[2]);
    auto sidx = CastVal(builder, loc, self->VisitExpr(call->args[3]), builder.getIndexType(), false);
    rhs = builder.create<mlir::pto::TGetValOp>(loc, dtype, stile, sidx).getResult();
  } else {
    rhs = self->VisitExpr(call->args[2]);
  }
  builder.create<OpTy>(loc, lhs, rhs, res);
}

template <typename OpTy>
static void EmitScalarBinaryOpWithFlag(CodeGenPTOAS *self, mlir::OpBuilder &builder, mlir::Location loc, const CallNode *call, bool flag) {
  auto res = self->GetAsTile(call->args[0]);
  auto lhs = self->GetAsTile(call->args[1]);
  auto rhs = self->VisitExpr(call->args[2]);
  builder.create<OpTy>(loc, lhs, rhs, res);
}

void CodeGenPTOAS::VisitStmt_(const LetStmtNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Value value = VisitExpr(op->value);
  if (!value) {
    LOG(FATAL) << "Failed to generate value for LetStmt: " << op->var->name_hint;
  }
  symbolTable[op->var.get()] = value;
  VisitStmt(op->body);
}

void CodeGenPTOAS::VisitStmt_(const AttrStmtNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  // Handle the attribute statement
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    mlir::Value rawIdx;
    if (iv->thread_tag == "blockIdx.x") {
      rawIdx = builder.create<mlir::pto::GetBlockIdxOp>(loc).getResult();
    } else if (iv->thread_tag == "blockIdx.y") {
      rawIdx = builder.create<mlir::pto::GetSubBlockIdxOp>(loc).getResult();
    } else {
      LOG(FATAL) << "Unsupported thread_tag: " << iv->thread_tag;
    }

    // Resolve desired MLIR type for the IterVar's var
    mlir::Type dstType = resolveArithType(iv->var.dtype());
    bool dstSigned = iv->var.dtype().is_int();

    mlir::Value idxVal = CastVal(builder, loc, rawIdx, dstType, dstSigned);
    if (!idxVal) {
      LOG(FATAL) << "Failed to convert block index value for " << iv->var->name_hint;
    }
    symbolTable[iv->var.get()] = idxVal;
    VisitStmt(op->body);
  } else if (op->attr_key == "resource_scope") {
    auto resource_id = Downcast<IntImm>(op->value)->value;
    auto resource_name = resource_id == 0 ? "CUBE" : "VEC";
    std::string old_scope = source_scope;

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      source_scope = resource_name;
      mlir::Region *region;
      if (resource_id == 0) {
        region = &builder.create<mlir::pto::SectionCubeOp>(loc).getBody();
      } else {
        region = &builder.create<mlir::pto::SectionVectorOp>(loc).getBody();
      }
      auto &block = region->emplaceBlock();
      builder.setInsertionPointToStart(&block);

      VisitStmt(op->body);
    }

    source_scope = old_scope;
  } else {
    LOG(FATAL) << "Unsupported attribute: " << op->attr_key;
  }
}

static std::pair<bool, int64_t> try_eval(const PrimExpr &shape) {
  if (const auto *c = shape.as<IntImmNode>()) {
    return {true, c->value};
  } else {
    return {false, -1};
  }
}

static int64_t eval(const PrimExpr &shape) {
  if (const auto *c = shape.as<IntImmNode>()) {
    return c->value;
  } else {
    LOG(FATAL) << "Only constant shapes are supported, got: " << shape;
    return -1;
  }
}

void CodeGenPTOAS::VisitStmt_(const AllocateNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode *buffer = op->buffer_var.as<VarNode>();
  auto type = resolveArithType(op->dtype);

  auto shape = buffer_shapes_[op->buffer_var];
  std::vector<int64_t> gshape;

  if (scope == "shared") {
    if (shape.size() == 2) {
      gshape.push_back(eval(shape[0]));
      gshape.push_back(eval(shape[1]));
      auto ub = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::VEC);
      auto bl = BLayoutAttr::get(&context, BLayout::RowMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::NoneBox);
      auto pd = PadValueAttr::get(&context, PadValue::Null);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 512);

      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, ub, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else if (shape.size() == 1) {
      gshape.push_back(1);
      gshape.push_back(eval(shape[0]));
      auto ub = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::VEC);
      auto bl = BLayoutAttr::get(&context, BLayout::RowMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::NoneBox);
      auto pd = PadValueAttr::get(&context, PadValue::Null);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 512);

      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, ub, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else {
      LOG(FATAL) << "Only 2D shared memory allocation is supported, got shape size: " << shape.size();
    }
  } else if (scope == "shared.dyn") {
    if (shape.size() == 2) {
      gshape.push_back(eval(shape[0]));
      gshape.push_back(eval(shape[1]));
      auto mat = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::MAT);
      auto bl = BLayoutAttr::get(&context, BLayout::ColMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::RowMajor);
      auto pd = PadValueAttr::get(&context, PadValue::Zero);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 512);

      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, mat, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else {
      LOG(FATAL) << "Only 2D dynamic shared memory allocation is supported, got shape size: " << shape.size();
    }
  } else if (scope == "wmma.accumulator") {
    if (shape.size() == 2) {
      gshape.push_back(eval(shape[0]));
      gshape.push_back(eval(shape[1]));
      auto acc = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::ACC);
      auto bl = BLayoutAttr::get(&context, BLayout::ColMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::RowMajor);
      auto pd = PadValueAttr::get(&context, PadValue::Null);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 1024);

      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, acc, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else {
      LOG(FATAL) << "Only 2D WMMA accumulator memory allocation is supported, got shape size: " << shape.size();
    }
  } else if (scope == "wmma.matrix_a") {
    if (shape.size() == 2) {
      gshape.push_back(eval(shape[0]));
      gshape.push_back(eval(shape[1]));
      auto left = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::LEFT);
      auto bl = BLayoutAttr::get(&context, BLayout::RowMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::RowMajor);
      auto pd = PadValueAttr::get(&context, PadValue::Null);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 512);

      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, left, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else {
      LOG(FATAL) << "Only 2D WMMA matrix A memory allocation is supported, got shape size: " << shape.size();
    }
  } else if (scope == "wmma.matrix_b") {
    if (shape.size() == 2) {
      gshape.push_back(eval(shape[0]));
      gshape.push_back(eval(shape[1]));
      auto right = mlir::pto::AddressSpaceAttr::get(&context, mlir::pto::AddressSpace::RIGHT);
      auto bl = BLayoutAttr::get(&context, BLayout::RowMajor);
      auto sl = SLayoutAttr::get(&context, SLayout::ColMajor);
      auto pd = PadValueAttr::get(&context, PadValue::Null);
      auto fractal = mlir::IntegerAttr::get(builder.getIntegerType(32), 512);
      auto cfg = mlir::pto::TileBufConfigAttr::get(&context, bl, sl, fractal, pd);
      auto tile = mlir::pto::TileBufType::get(&context, gshape, type, right, gshape, cfg);
      auto alloca = builder.create<mlir::pto::AllocTileOp>(loc, tile, mlir::Value(), mlir::Value());
      symbolTable[buffer] = alloca.getResult();
    } else {
      LOG(FATAL) << "Only 2D WMMA matrix B memory allocation is supported, got shape size: " << shape.size();
    }
  } else {
    LOG(FATAL) << "Unsupported allocation scope: " << scope;
  }
  VisitStmt(op->body);
}

void CodeGenPTOAS::VisitStmt_(const SeqStmtNode *op) {
  for (const Stmt &stmt : op->seq) {
    this->VisitStmt(stmt);
  }
}

void CodeGenPTOAS::VisitStmt_(const EvaluateNode *op) {
  // evaluate expression and ignore its result
  VisitExpr(op->value);
}

static void GenLoopUnroll(CodeGenPTOAS *self, const ForNode *op) {
  mlir::Location loc = self->builder.getUnknownLoc();
  int64_t min = eval(op->min);
  int64_t extent = eval(op->extent);
  for (int64_t i = 0; i < extent; ++i) {
    mlir::Value loopVar = self->builder.create<mlir::arith::ConstantOp>(
        loc,
        self->resolveArithType(op->loop_var.dtype()),
        self->builder.getIntegerAttr(self->resolveArithType(op->loop_var.dtype()), min + i)).getResult();
    self->symbolTable[op->loop_var.get()] = loopVar;
    self->VisitStmt(op->body);
  }
}

static void GenLoopNormal(CodeGenPTOAS *self, const ForNode *op) {
  mlir::Location loc = self->builder.getUnknownLoc();
  mlir::Value lb = CastVal(self->builder, loc, self->VisitExpr(op->min), self->builder.getIndexType(), false);
  mlir::Value ext = CastVal(self->builder, loc, self->VisitExpr(op->extent), self->builder.getIndexType(), false);
  if (!lb || !ext) {
    LOG(FATAL) << "Failed to generate loop bounds for for-loop.";
  }
  auto one = self->builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();
  // ub = lb + extent
  mlir::Value ub = self->builder.create<mlir::arith::AddIOp>(loc, lb, ext).getResult();

  auto forOp = self->builder.create<mlir::scf::ForOp>(loc, lb, ub, one);

  // Emit loop body into the created region, map induction var to expected arith-legal type.
  {
    mlir::OpBuilder::InsertionGuard guard(self->builder);
    auto &body = forOp.getRegion().front();
    self->builder.setInsertionPointToStart(&body);

    // induction var is body.getArgument(0) (index type). Cast to the loop_var's dtype.
    mlir::Type dstType = self->resolveArithType(op->loop_var.dtype());
    bool dstSigned = op->loop_var.dtype().is_int();
    mlir::Value iv_index = body.getArgument(0);
    mlir::Value iv_cast = CastVal(self->builder, loc, iv_index, dstType, dstSigned);
    if (!iv_cast) {
      LOG(FATAL) << "Failed to cast induction variable for " << op->loop_var->name_hint;
    }
    self->symbolTable[op->loop_var.get()] = iv_cast;

    // Emit loop body
    self->VisitStmt(op->body);
  }
}

void CodeGenPTOAS::VisitStmt_(const ForNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  if (op->kind == tir::ForKind::kSerial) {
#if 0
    GenLoopUnroll(this, op);
#else
    GenLoopNormal(this, op);
#endif
  } else {
    LOG(FATAL) << "Only serial for-loops are supported.";
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const BufferLoadNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  std::string scope = op->buffer.scope();
  auto sym = symbolTable[op->buffer->data.get()];
  auto off = CastVal(builder, loc, VisitExpr(op->indices.back()), builder.getIndexType(), false);
  auto dtype = resolveArithType(op->dtype);
  if (scope == "" || scope == "global") {
    LOG(FATAL) << "Global memory access is not supported yet.";
  } else {
    return builder.create<mlir::pto::TGetValOp>(loc, dtype, sym, off).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const IntImmNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type type = builder.getIntegerType(op->dtype.bits());
  mlir::IntegerAttr intAttr = builder.getIntegerAttr(type, op->value);
  return builder.create<mlir::arith::ConstantOp>(loc, type, intAttr).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const VarNode *op) {
  if (symbolTable.count(op)) {
    return symbolTable.at(op);
  } else {
    LOG(FATAL) << "Undefined variable: " << op->name_hint;
    return nullptr;
  }
}

static mlir::pto::PIPE GetPipe(const std::string &pipe_name) {
  if (pipe_name == "ALL") {
    return mlir::pto::PIPE::PIPE_ALL;
  } else if (pipe_name == "MTE1") {
    return mlir::pto::PIPE::PIPE_MTE1;
  } else if (pipe_name == "MTE2") {
    return mlir::pto::PIPE::PIPE_MTE2;
  } else if (pipe_name == "MTE3") {
    return mlir::pto::PIPE::PIPE_MTE3;
  } else if (pipe_name == "MTE4") {
    return mlir::pto::PIPE::PIPE_MTE4;
  } else if (pipe_name == "MTE5") {
    return mlir::pto::PIPE::PIPE_MTE5;
  } else if (pipe_name == "S") {
    return mlir::pto::PIPE::PIPE_S;
  } else if (pipe_name == "V") {
    return mlir::pto::PIPE::PIPE_V;
  } else if (pipe_name == "M") {
    return mlir::pto::PIPE::PIPE_M;
  } else if (pipe_name == "V2") {
    return mlir::pto::PIPE::PIPE_V2;
  } else if (pipe_name == "FIX") {
    return mlir::pto::PIPE::PIPE_FIX;
  } else if (pipe_name == "L1A") {
    return mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1A;
  } else if (pipe_name == "L1B") {
    return mlir::pto::PIPE::VIRTUAL_PIPE_MTE2_L1B;
  } else if (pipe_name == "NUM") {
    return mlir::pto::PIPE::PIPE_NUM;
  } else {
    LOG(FATAL) << "Unsupported pipe name: " << pipe_name;
  }
}

mlir::Value CodeGenPTOAS::GetAsTile(const PrimExpr &op) {
  ICHECK(op.as<CallNode>()->op.same_as(builtin::tvm_access_ptr())) << "Illegal tile descriptor.";
  auto buffer = op.as<CallNode>()->args[1].as<VarNode>();
  auto offset = op.as<CallNode>()->args[2];
  auto [succ, offval] = try_eval(offset);
  if (offval != 0) {
    auto src = symbolTable.at(buffer);
    auto loc = builder.getUnknownLoc();
    auto ttype = llvm::cast<mlir::pto::TileBufType>(src.getType());
    auto sizes = ttype.getShape();
    auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    std::vector offsets(sizes.size(), zero);
    offsets.back() = CastVal(builder, loc, VisitExpr(offset), builder.getIndexType(), false);
    auto tile = builder.create<mlir::pto::SubsetOp>(loc, src, offsets, builder.getI64ArrayAttr(sizes)).getResult();
    return tile;
  }
  ICHECK(symbolTable.count(buffer) && symbolTable.at(buffer)) << "undefined tile buffer.";
  return symbolTable.at(buffer);
}

mlir::Value CodeGenPTOAS::VisitExpr_(const CallNode *op) {
  if (op->op.same_as(builtin::call_extern())) {
    return CallExternCodegen(op);
  }

  mlir::Location loc = builder.getUnknownLoc();

  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    auto pipe = GetPipe(Downcast<StringImm>(op->args[0])->value);
    builder.create<mlir::pto::BarrierOp>(loc, mlir::pto::PipeAttr::get(&context, pipe));
    return mlir::Value();
  }

  if (op->op.same_as(tl::ascend_wait_cross_flag())) {
    auto flag = Downcast<IntImm>(op->args[0])->value;
    auto pstr = Downcast<StringImm>(op->args[1])->value;
    mlir::pto::PIPE pipe;
    if (pstr.empty()) {
      if (this->source_scope == "CUBE") {
        pipe = mlir::pto::PIPE::PIPE_MTE1;
      } else if (this->source_scope == "VEC") {
        pipe = mlir::pto::PIPE::PIPE_V;
      } else {
        LOG(FATAL) << "Unknown source scope for wait_cross_flag: " << this->source_scope;
      }
    } else {
      LOG(FATAL) << "Specifying pipe name in wait_cross_flag is not supported yet.";
    }
    builder.create<mlir::pto::SyncWaitOp>(loc, mlir::pto::PipeAttr::get(&context, pipe), mlir::IntegerAttr::get(builder.getIntegerType(32), flag));
    return mlir::Value();
  }

  if (op->op.same_as(tl::ascend_set_cross_flag())) {
    auto pstr = Downcast<StringImm>(op->args[0])->value;
    auto flag = Downcast<IntImm>(op->args[1])->value;
    mlir::pto::PIPE pipe = GetPipe(pstr);
    builder.create<mlir::pto::SyncSetOp>(loc, mlir::pto::PipeAttr::get(&context, pipe), mlir::IntegerAttr::get(builder.getIntegerType(32), flag));
    return mlir::Value();
  }

  if (op->op.same_as(tl::ascend_fill())) {
    auto res = GetAsTile(op->args[1]);
    auto value = VisitExpr(op->args[2]);
    builder.create<mlir::pto::TExpandsOp>(loc, value, res);
    return mlir::Value();
  }

  if (op->op.same_as(tl::ascend_gemm_v0())) {
    auto a = GetAsTile(op->args[1]);
    auto b = GetAsTile(op->args[2]);
    auto c = GetAsTile(op->args[3]);
    auto clear = eval(op->args[4]);
    auto dtype = llvm::cast<mlir::pto::TileBufType>(c.getType()).getElementType();
    auto restype = mlir::RankedTensorType::get(mlir::cast<mlir::pto::TileBufType>(c.getType()).getShape(), dtype);
    if (clear)
      builder.create<mlir::pto::TMatmulOp>(loc, restype, a, b, mlir::Value(), c);
    else
      builder.create<mlir::pto::TMatmulAccOp>(loc, restype, c, a, b, c);
    return mlir::Value();
  }

  // unary
  if (op->op.same_as(tl::ascend_exp())) {
    EmitUnaryOp<mlir::pto::TExpOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_ln())) {
    EmitUnaryOp<mlir::pto::TLogOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_abs())) {
    EmitUnaryOp<mlir::pto::TAbsOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_reciprocal())) {
    EmitUnaryOp<mlir::pto::TRecipOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_sqrt())) {
    EmitUnaryOp<mlir::pto::TSqrtOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_rsqrt())) {
    EmitUnaryOp<mlir::pto::TRsqrtOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_relu())) {
    EmitUnaryOp<mlir::pto::TReluOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_bitwise_not())) {
    EmitUnaryOp<mlir::pto::TNotOp>(this, builder, loc, op); return mlir::Value();
  }

  // binary (with flag where applicable)
  if (op->op.same_as(tl::ascend_add())) {
    EmitBinaryOp<mlir::pto::TAddOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_sub())) {
    EmitBinaryOp<mlir::pto::TSubOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_mul())) {
    EmitBinaryOp<mlir::pto::TMulOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_div())) {
    EmitBinaryOp<mlir::pto::TDivOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_max())) {
    EmitBinaryOp<mlir::pto::TMaxOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_min())) {
    EmitBinaryOp<mlir::pto::TMinOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_bitwise_and())) {
    EmitBinaryOp<mlir::pto::TAndOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_bitwise_or())) {
    EmitBinaryOp<mlir::pto::TOrOp>(this, builder, loc, op); return mlir::Value();
  }

  // scalar/broadcast variants
  if (op->op.same_as(tl::ascend_adds())) {
    EmitScalarBinaryOp<mlir::pto::TAddSOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_subs())) {
    EmitScalarBinaryOp<mlir::pto::TSubSOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_muls())) {
    EmitScalarBinaryOp<mlir::pto::TMulsOp>(this, builder, loc, op); return mlir::Value();
  }
  if (op->op.same_as(tl::ascend_divs())) {
    EmitScalarBinaryOp<mlir::pto::TDivSOp>(this, builder, loc, op); return mlir::Value();
  }

  // reduce variants
  if (op->op.same_as(tl::ascend_reduce())) {
    std::string opname = Downcast<StringImm>(op->args[0])->value;
    auto templates = ExtractTemplateParamsForSliceBuffer(opname);
    auto rdim = std::get<2>(templates);
    auto res = GetAsTile(op->args[1]);
    auto src = GetAsTile(op->args[2]);
    if (rdim == -1) {
      if (opname.find("reduce_sum") != std::string::npos) {
        builder.create<mlir::pto::TRowSumOp>(loc, src, res);
        return mlir::Value();
      } else if (opname.find("reduce_max") != std::string::npos) {
        builder.create<mlir::pto::TRowMaxOp>(loc, src, res);
        return mlir::Value();
      } else {
        LOG(FATAL) << "Unsupported reduce operation: " << opname;
      }
    } else {
      LOG(FATAL) << "Unsupported reduce dimension: " << rdim;
    }
  }

  LOG(FATAL) << "Unsupported call operation: " << op->op;
  return nullptr;
}

mlir::Value CodeGenPTOAS::VisitExpr_(const FloatImmNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type type = resolveArithType(op->dtype);
  mlir::FloatAttr fattr = mlir::FloatAttr::get(type, op->value);
  return builder.create<mlir::arith::ConstantOp>(loc, type, fattr).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const StringImmNode *op) {
  LOG(FATAL) << "StringImm used where an MLIR value is expected: " << op->value;
  return nullptr;
}

mlir::Value CodeGenPTOAS::VisitExpr_(const CastNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto val = VisitExpr(op->value);
  mlir::Type dst = resolveArithType(op->dtype);
  // If same type, just pass through.
  if (val.getType() == dst) return val;

  bool dst_is_float = op->dtype.is_float();
  bool src_is_float = op->value.dtype().is_float();
  if (src_is_float && !dst_is_float) {
    // float -> int
    if (op->value.dtype().is_int()) {
      return builder.create<mlir::arith::FPToSIOp>(loc, dst, val).getResult();
    } else {
      return builder.create<mlir::arith::FPToUIOp>(loc, dst, val).getResult();
    }
  } else if (!src_is_float && dst_is_float) {
    // int -> float
    if (op->value.dtype().is_int()) {
      return builder.create<mlir::arith::SIToFPOp>(loc, dst, val).getResult();
    } else {
      return builder.create<mlir::arith::UIToFPOp>(loc, dst, val).getResult();
    }
  } else if (!src_is_float && !dst_is_float) {
    // integer -> integer (extend/trunc)
    int src_bits = op->value.dtype().bits();
    int dst_bits = op->dtype.bits();
    if (dst_bits > src_bits) {
      if (op->value.dtype().is_int()) {
        return builder.create<mlir::arith::ExtSIOp>(loc, dst, val).getResult();
      } else {
        return builder.create<mlir::arith::ExtUIOp>(loc, dst, val).getResult();
      }
    } else {
      return builder.create<mlir::arith::TruncIOp>(loc, dst, val).getResult();
    }
  } else {
    // float -> float (ext / trunc)
    int src_bits = op->value.dtype().bits();
    int dst_bits = op->dtype.bits();
    if (dst_bits > src_bits) {
      return builder.create<mlir::arith::ExtFOp>(loc, dst, val).getResult();
    } else {
      return builder.create<mlir::arith::TruncFOp>(loc, dst, val).getResult();
    }
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const AddNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::AddFOp>(loc, a, b).getResult();
  } else {
    return builder.create<mlir::arith::AddIOp>(loc, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const SubNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::SubFOp>(loc, a, b).getResult();
  } else {
    return builder.create<mlir::arith::SubIOp>(loc, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const MulNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::MulFOp>(loc, a, b).getResult();
  } else {
    return builder.create<mlir::arith::MulIOp>(loc, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const DivNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::DivFOp>(loc, a, b).getResult();
  } else {
    if (op->dtype.is_uint()) {
      return builder.create<mlir::arith::DivUIOp>(loc, a, b).getResult();
    } else {
      return builder.create<mlir::arith::DivSIOp>(loc, a, b).getResult();
    }
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const ModNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::RemFOp>(loc, a, b).getResult();
  } else {
    if (op->dtype.is_uint()) {
      return builder.create<mlir::arith::RemUIOp>(loc, a, b).getResult();
    } else {
      return builder.create<mlir::arith::RemSIOp>(loc, a, b).getResult();
    }
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const MinNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    // min(a,b) -> select(a < b, a, b)
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, a, b).getResult();
    return builder.create<mlir::arith::SelectOp>(loc, cmp, a, b).getResult();
  } else {
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, a, b).getResult();
    return builder.create<mlir::arith::SelectOp>(loc, cmp, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const MaxNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, a, b).getResult();
    return builder.create<mlir::arith::SelectOp>(loc, cmp, a, b).getResult();
  } else {
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, a, b).getResult();
    return builder.create<mlir::arith::SelectOp>(loc, cmp, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const LTNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, a, b).getResult();
  } else {
    return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const LENode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, a, b).getResult();
  } else {
    return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const GTNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, a, b).getResult();
  } else {
    return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const GENode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, a, b).getResult();
  } else {
    return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const EQNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, a, b).getResult();
  } else {
    return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, a, b).getResult();
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const NENode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto eq = VisitExpr_(op->a.template as<EQNode>()); // fall back: build eq then xor
  // safer explicit compute:
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  if (op->dtype.is_float()) {
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, a, b).getResult();
    return cmp;
  } else {
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, a, b).getResult();
    return cmp;
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const AndNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  return builder.create<mlir::arith::AndIOp>(loc, a, b).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const OrNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();
  return builder.create<mlir::arith::OrIOp>(loc, a, b).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const NotNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto v = VisitExpr(op->a);
  if (!v) return mlir::Value();
  // Compute xor with 1 for boolean/integer negation
  auto one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getIntegerType(1), 1)).getResult();
  return builder.create<mlir::arith::XOrIOp>(loc, v, one).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const SelectNode *op) {
  mlir::Location loc = builder.getUnknownLoc();
  auto cond = VisitExpr(op->condition);
  auto t = VisitExpr(op->true_value);
  auto f = VisitExpr(op->false_value);
  if (!cond || !t || !f) return mlir::Value();
  return builder.create<mlir::arith::SelectOp>(loc, cond, t, f).getResult();
}

mlir::Value CodeGenPTOAS::VisitExpr_(const LetNode *op) {
  // Evaluate value, bind to symbol, evaluate body, then restore previous binding.
  auto varnode = op->var.as<VarNode>();
  mlir::Value v = VisitExpr(op->value);
  if (!v) return mlir::Value();
  mlir::Value prev = nullptr;
  if (symbolTable.count(varnode)) {
    prev = symbolTable[varnode];
  }
  symbolTable[varnode] = v;
  mlir::Value res = VisitExpr(op->body);
  if (prev) {
    symbolTable[varnode] = prev;
  } else {
    symbolTable.erase(varnode);
  }
  return res;
}

mlir::Value CodeGenPTOAS::VisitExpr_(const FloorDivNode *op) {
  // Inputs are guaranteed positive => floor(a/b) == trunc(a/b) for integers.
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();

  if (op->dtype.is_float()) {
    // float floor-div: use floating division (for positive values this is fine)
    return builder.create<mlir::arith::DivFOp>(loc, a, b).getResult();
  } else {
    // integers (positive): truncating integer division equals floor division
    if (op->dtype.is_uint()) {
      return builder.create<mlir::arith::DivUIOp>(loc, a, b).getResult();
    } else {
      return builder.create<mlir::arith::DivSIOp>(loc, a, b).getResult();
    }
  }
}

mlir::Value CodeGenPTOAS::VisitExpr_(const FloorModNode *op) {
  // For positive inputs, floormod reduces to the usual remainder.
  mlir::Location loc = builder.getUnknownLoc();
  auto a = VisitExpr(op->a);
  auto b = VisitExpr(op->b);
  if (!a || !b) return mlir::Value();

  if (op->dtype.is_float()) {
    // floating-point remainder
    return builder.create<mlir::arith::RemFOp>(loc, a, b).getResult();
  } else {
    if (op->dtype.is_uint()) {
      return builder.create<mlir::arith::RemUIOp>(loc, a, b).getResult();
    } else {
      return builder.create<mlir::arith::RemSIOp>(loc, a, b).getResult();
    }
  }
}

static std::vector<mlir::Value> legalShape(CodeGenPTOAS *codegen, mlir::Location loc, const std::vector<PrimExpr> &shape) {
  std::vector<mlir::Value> legalShape;
  for (auto dim : shape) {
    legalShape.push_back(CastVal(codegen->builder, loc, codegen->VisitExpr(dim), codegen->builder.getIndexType(), false));
  }
  return legalShape;
}

static std::vector<mlir::Value> computeNDStrides(mlir::OpBuilder &builder, mlir::Location loc, const std::vector<mlir::Value> &shape) {
  int n = shape.size();
  std::vector<mlir::Value> strides(n);

  if (n == 0) return strides;

  mlir::Type indexType = builder.getIndexType();
  mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();

  strides[n - 1] = one;

  for (int i = n - 2; i >= 0; --i) {
    mlir::Value prevStride = strides[i + 1];
    mlir::Value dimSize = shape[i + 1];
    strides[i] = builder.create<mlir::arith::MulIOp>(loc, prevStride, dimSize).getResult();
  }
  return strides;
}

static std::vector<mlir::Value> makeSubviewSizes(mlir::OpBuilder &builder, mlir::Location loc, llvm::ArrayRef<int64_t> shape, size_t num) {
  std::vector<mlir::Value> sizes;
  size_t high = num - shape.size();
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();
  for (size_t i = 0; i < high; ++i) {
    sizes.push_back(one);
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    auto val = builder.create<mlir::arith::ConstantIndexOp>(loc, shape[i]).getResult();
    sizes.push_back(val);
  }
  return sizes;
}

mlir::Value CodeGenPTOAS::CallExternCodegen(const CallNode *op) {
  std::string op_name = Downcast<StringImm>(op->args[0])->value;
  auto loc = builder.getUnknownLoc();
  if (op_name.find("tl::ascend::copy") != std::string::npos) {
    auto src_var = op->args[1].as<CallNode>()->args[1].as<VarNode>();
    auto dst_var = op->args[2].as<CallNode>()->args[1].as<VarNode>();

    auto src_offset = VisitExpr(op->args[1].as<CallNode>()->args[2]);
    auto dst_offset = VisitExpr(op->args[2].as<CallNode>()->args[2]);

    auto src_len = VisitExpr(op->args[1].as<CallNode>()->args[3]);
    auto dst_len = VisitExpr(op->args[2].as<CallNode>()->args[3]);

    auto src_type = op->args[1].as<CallNode>()->args[0].as<CallNode>()->dtype;
    auto dst_type = op->args[2].as<CallNode>()->args[0].as<CallNode>()->dtype;

    if (op_name.find("copy_gm_to_ub") != std::string::npos || op_name.find("copy_gm_to_l1") != std::string::npos) {
      size_t op_arg_len = op->args.size();
      long cols = eval(op->args[op_arg_len - 1]);
      long rows = op_arg_len == 5 ? 1 : eval(op->args[op_arg_len - 2]);
      auto ptype = llvm::cast<mlir::pto::PtrType>(symbolTable[src_var].getType());
      auto dtype = ptype.getElementType();
      auto gbase = symbolTable[src_var];
      auto gshape = legalShape(this, loc, bufferTable[src_var].shape);
      auto gviewtype = mlir::pto::TensorViewType::get(&context, gshape.size(), dtype);
      ICHECK(gshape.size() != 0) << "gm shape cannot be empty.";
      auto gstride = computeNDStrides(builder, loc, gshape);
      auto gview = builder.create<mlir::pto::MakeTensorViewOp>(loc, gviewtype, gbase, gshape, gstride, nullptr).getResult();
      auto subviewtype = mlir::pto::PartitionTensorViewType::get(&context, {rows, cols}, dtype);
      auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
      std::vector<mlir::Value> offsets(gshape.size(), zero);
      offsets.back() = CastVal(builder, loc, src_offset, builder.getIndexType(), false);
      auto sizes = makeSubviewSizes(builder, loc, {rows, cols}, gshape.size());
      auto gsubview = builder.create<mlir::pto::PartitionViewOp>(loc, subviewtype, gview, offsets, sizes).getResult();
      auto restype = mlir::RankedTensorType::get({rows, cols}, dtype);
      auto tload = builder.create<mlir::pto::TLoadOp>(loc, restype, gsubview, symbolTable[dst_var]);
      return tload.getResult();
    } else if (op_name.find("copy_ub_to_gm") != std::string::npos || op_name.find("copy_l0c_to_gm") != std::string::npos) {
      size_t op_arg_len = op->args.size();
      long cols = eval(op->args[op_arg_len - 1]);
      long rows = op_arg_len == 5 ? 1 : eval(op->args[op_arg_len - 2]);
      auto ptype = llvm::cast<mlir::pto::PtrType>(symbolTable[dst_var].getType());
      auto dtype = ptype.getElementType();
      auto gbase = symbolTable[dst_var];
      auto gshape = legalShape(this, loc, bufferTable[dst_var].shape);
      auto gviewtype = mlir::pto::TensorViewType::get(&context, gshape.size(), dtype);
      ICHECK(gshape.size() != 0) << "gm shape cannot be empty.";
      auto gstride = computeNDStrides(builder, loc, gshape);
      auto gview = builder.create<mlir::pto::MakeTensorViewOp>(loc, gviewtype, gbase, gshape, gstride, nullptr).getResult();
      auto subviewtype = mlir::pto::PartitionTensorViewType::get(&context, {rows, cols}, dtype);
      auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
      std::vector<mlir::Value> offsets(gshape.size(), zero);
      offsets.back() = CastVal(builder, loc, dst_offset, builder.getIndexType(), false);
      auto sizes = makeSubviewSizes(builder, loc, {rows, cols}, gshape.size());
      auto gsubview = builder.create<mlir::pto::PartitionViewOp>(loc, subviewtype, gview, offsets, sizes).getResult();
      auto restype = mlir::RankedTensorType::get({rows, cols}, dtype);
      builder.create<mlir::pto::TStoreOp>(loc, restype, symbolTable[src_var], gsubview);
      return mlir::Value();
    } else if (op_name.find("copy_ub_to_ub") != std::string::npos) {
      auto res = GetAsTile(op->args[2]);
      auto src = GetAsTile(op->args[1]);
      if (src_type == dst_type) {
        auto dtype = llvm::cast<mlir::pto::TileBufType>(res.getType()).getElementType();
        auto restype = mlir::RankedTensorType::get(mlir::cast<mlir::pto::TileBufType>(res.getType()).getShape(), dtype);
        builder.create<mlir::pto::TMovOp>(loc, restype, src, res);
      } else {
        builder.create<mlir::pto::TCvtOp>(loc, src, res);
      }
      return mlir::Value();
    } else {
      LOG(FATAL) << "Unsupported copy operation: " << op_name;
    }
  } else {
    LOG(FATAL) << "Unsupported call operation: " << op_name;
  }
}

void CodeGenPTOAS::UbShapeInputCheck(const AllocateNode *op) {
  auto shape = buffer_shapes_[op->buffer_var];
  if (shape.size() > 3 || shape.size() == 0){
    ICHECK(false) << "Unsupported ubsize which is expected to be 1, 2 or 3";
  }
}

bool CodeGenPTOAS::ValidLayoutEnabled(const AllocateNode *op) {
  auto shape = buffer_shapes_[op->buffer_var];
  bool valid = false;
  int8_t typeSize = op->dtype.bits() / 8;
  if (tvm::tir::is_zero(tvm::truncmod(shape[1] * typeSize, 32))) {
    valid = false;
  } else {
    valid = true;
  }
  return valid;
}

void CodeGenPTOAS::AddFunction(const GlobalVar &gvar, const PrimFunc &func) {
  address_map_ = func->GetAttr<Map<Var, PrimExpr>>("address_map").value_or(Map<Var, PrimExpr>());
  buffer_shapes_ = func->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess").value_or(Map<Var, Array<PrimExpr>>());

  builder.setInsertionPointToEnd(module->getBody());
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Type, 4> argTypes;
  llvm::SmallVector<const tir::VarNode *, 4> shapeVals;

  for (const auto &param : func->params) {
    // 1. Check buffer_map first for high-level buffer info
    if (func->buffer_map.count(param)) {
      Buffer buf = func->buffer_map.at(param);
      mlir::Type elemType = resolveArithType(buf->dtype);
      argTypes.push_back(mlir::pto::PtrType::get(&context, elemType));

      for (size_t i = 0; i < buf->shape.size(); ++i) {
        auto dim = buf->shape[i].as<VarNode>();
        if (!dim)
          continue;
        if (std::find(shapeVals.begin(), shapeVals.end(), dim) == shapeVals.end()) {
          shapeVals.push_back(dim);
        }
      }
    }
    // 2. Otherwise, look at the Var's type_annotation for nested pointers
    else if (param->type_annotation.defined()) {
      argTypes.push_back(resolveType(param->type_annotation));
    }
    // 3. Fallback to basic DataType
    else {
      argTypes.push_back(resolveArithType(param->dtype));
    }
  }

  for (auto dim : shapeVals) {
    argTypes.push_back(resolveArithType(dim->dtype));
  }

  std::string funcName = gvar->name_hint.operator std::string();
  auto funcType = builder.getFunctionType(argTypes, {});
  auto funcOp = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);

  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  for (size_t i = 0; i < func->params.size(); ++i) {
    auto param = func->params[i];
    if (func->buffer_map.count(param)) {
      Buffer buf = func->buffer_map.at(param);
      // Buffer parameter
      param = buf->data;
      // Record buffer shape information
      for (size_t i = 0; i < buf->shape.size(); ++i) {
        auto dim = buf->shape[i];
        bufferTable[buf->data.get()].shape.push_back(dim);
      }
    }
    mlir::Value mlir_param = entryBlock->getArgument(i);
    symbolTable[param.get()] = mlir_param;
  }

  for (size_t i = 0; i < shapeVals.size(); ++i) {
    mlir::Value mlir_param = entryBlock->getArgument(func->params.size() + i);
    symbolTable[shapeVals[i]] = mlir_param;
  }

  this->VisitStmt(func->body);

  if (entryBlock->empty() || !entryBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    builder.create<mlir::func::ReturnOp>(loc);
  }
}

} // namespace codegen
} // namespace tvm

namespace tvm {
namespace codegen {

CodeGenTileLangPTOAS::CodeGenTileLangPTOAS(std::string platform) {
}

std::string CodeGenTileLangPTOAS::Finish() {
  return super::Finish();
}

} // namespace codegen
} // namespace tvm
