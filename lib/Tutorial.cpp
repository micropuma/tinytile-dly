#include "Tutorial.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "Tutorial.cpp.inc"

void mlir::tutorial::TutorialDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tutorial.cpp.inc"
      >();
}

#include "TutorialDialect.cpp.inc"

using namespace mlir;
using namespace mlir::tutorial;

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Operation *getSlice(OpBuilder &b, Location loc, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) -> Operation * { return nullptr; });
}

// 获取DequantOp操作的迭代区间
SmallVector<Range> DequantOp::getIterationDomain(OpBuilder &b) {
  int64_t rank = getInput().getType().getRank();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, getLoc(), getInput());

  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = sizes[dim];
    loopBounds[dim].stride = one;
  }

  return loopBounds;
}

// 获取DequantOp操作的迭代器类型
SmallVector<utils::IteratorType> DequantOp::getLoopIteratorTypes() {
  int64_t rank = getInput().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

LogicalResult DequantOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

// 核心操作，实现DequantOp的tiling
FailureOr<TilingResult> DequantOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  int64_t rank = getInput().getType().getRank();
  // 在dequant操作中，默认stride都是1
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  auto inputTile = b.create<tensor::ExtractSliceOp>(loc, getInput(), offsets,
                                                    sizes, strides);
  auto scaleTile = b.create<tensor::ExtractSliceOp>(loc, getScale(), offsets,
                                                    sizes, strides);

  Type resultType = inputTile.getResultType();

  Operation *tiledOp =
      mlir::clone(b, getOperation(), {resultType}, {inputTile, scaleTile});

  return TilingResult{{tiledOp},
                      SmallVector<Value>(tiledOp->getResults()),
                      {inputTile, scaleTile}};
}

// 为了计算结果的offsets和sizes，返回iteration domain的相应坐标
LogicalResult DequantOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // 先由结果坐标获取iteration domain的坐标，然后调用getTiledImplementation完成tiling
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

// 有目前的输入坐标，获取结果iteration domain
LogicalResult DequantOp::getIterationDomainTileFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  // dequant操作比较简单，就是1:1 mapping即可
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::getTiledImplementationFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromOperandTile(
          b, operandNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

// BufferizableOpInterface的实现
bool DequantOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  // 判断这个 operand 是否会被 buffer 读取
  return false;
}

bool DequantOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  // 判断这个 operand 是否会被 buffer 写入
  return false;
}

bufferization::AliasingValueList DequantOp::getAliasingValues(
    OpOperand &opOperand, const mlir::bufferization::AnalysisState &state) 
{
  return {};
}

LogicalResult DequantOp::bufferize(
    RewriterBase &rewriter, const bufferization::BufferizationOptions &state) {
  // 这里是真正的 bufferize 操作，把 tensor -> memref
  // TODO(leon): 如何将dequant操作land到memref层级上的composed-op
  return success();
}

// ========================== Tiling Op的实现示例 ==========================
// 实现ReluOp的tiling接口
// 获取ReluOp操作的迭代区间
SmallVector<Range> ReluOp::getIterationDomain(OpBuilder &b) {
  int64_t rank = getInput().getType().getRank();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, getLoc(), getInput());

  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = sizes[dim];
    loopBounds[dim].stride = one;
  }

  return loopBounds;
}

// 获取ReluOp操作的迭代器类型
// reluop是逐元素类型，所有维度均为parallel
SmallVector<utils::IteratorType> ReluOp::getLoopIteratorTypes() {
  int64_t rank = getInput().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

LogicalResult ReluOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

// 核心操作，实现ReluOp的tiling
FailureOr<TilingResult> ReluOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  int64_t rank = getInput().getType().getRank();
  // 在dequant操作中，默认stride都是1
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  // 去除一小块tile做计算
  auto inputTile = b.create<tensor::ExtractSliceOp>(loc, getInput(), offsets,
                                                    sizes, strides);

  Type resultType = inputTile.getResultType();

  Operation *tiledOp =
      mlir::clone(b, getOperation(), {resultType}, {inputTile});

  return TilingResult{{tiledOp},
                      SmallVector<Value>(tiledOp->getResults()),
                      {inputTile}};
}

// 为了计算结果的offsets和sizes，返回iteration domain的相应坐标
LogicalResult ReluOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

// 根据结果推断如何做tiling，符合将consumer tile好，fuse producer into consumer
FailureOr<TilingResult> ReluOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // 先由结果坐标获取iteration domain的坐标，然后调用getTiledImplementation完成tiling
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  // 获取tiling后的consumer的offsets & size map
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

// 从目前的输入坐标，获取结果iteration domain
LogicalResult ReluOp::getIterationDomainTileFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  // relu操作比较简单，就是1:1 mapping即可
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

// 由输入推断op如何做tiling
// 符合将producer tile好，fuse producer into consumer，给consumer做相应的tiling
FailureOr<TilingResult> ReluOp::getTiledImplementationFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromOperandTile(
          b, operandNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

// ========================== DPS + Tiling + BufferizableOpInterface的实现示例 ==========================
LogicalResult ReluOpDPS::verify() {
  // 注意，relu目前要求必须是ranked tensor类型
  // 照抄Linalg_softmaxOp的verifier实现
  ShapedType inputType = getInputOperandType();
  ShapedType outputType = getOutputOperandType();

  // TODO(leon): 为什么使用shapedtype而不是直接使用rankedtensor？
  if (!inputType.hasRank() || !outputType.hasRank()) {
    return emitOpError("input and output must be ranked tensor types");
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(inputShape, outputShape)))
    return emitOpError("incompatible output shape");
  return success();
}

SmallVector<Range> ReluOpDPS::getIterationDomain(OpBuilder &b) {
  int64_t rank = getInputOperandRank();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, getLoc(), getInput());

  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = sizes[dim];
    loopBounds[dim].stride = one;
  }

  return loopBounds;
}

SmallVector<utils::IteratorType> ReluOpDPS::getLoopIteratorTypes() {
  int64_t rank = getInputOperandRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

LogicalResult ReluOpDPS::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

// 注意这里和非DPS版本的区别
// 而且为了后续的bufferization，还需要同时考虑tensor type和memref type
// TODO(leon): 重构支持tensor和bufferization
FailureOr<TilingResult> ReluOpDPS::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputOperandRank();
  SmallVector<Value> tiledOperands;
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));

  Operation *inputSlice =
    getSlice(builder, getLoc(), getInput(), offsets, sizes, strides);
  tiledOperands.push_back(inputSlice->getResult(0));
  // DPS模式下，还需要对output做slice
  Operation *outputSlice =
    getSlice(builder, getLoc(), getOutput(), offsets, sizes, strides);
  tiledOperands.push_back(outputSlice->getResult(0));

  SmallVector<Type> resultTypes;
  // DestinationStypeOpInterface提供的接口
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledOp},
                      SmallVector<Value>(tiledOp->getResults()),
                      {inputSlice, outputSlice}};
}

// 为了计算结果的offsets和sizes，返回iteration domain的相应坐标
LogicalResult ReluOpDPS::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

// 根据结果推断如何做tiling，符合将consumer tile好，fuse producer into consumer
FailureOr<TilingResult> ReluOpDPS::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

LogicalResult ReluOpDPS::getIterationDomainTileFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> ReluOpDPS::getTiledImplementationFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromOperandTile(
          b, operandNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

// memory effects interface  
void tutorial::ReluOpDPS::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  
  // 1. 如果操作的是 Tensor，我们要模仿 Pure 的行为
  //    但是在 MemoryEffects 接口中，如果没有任何 Effect 被添加，
  //    它就默认被视为无副作用 (Pure)。
  //    只有当我们操作 MemRef 时，才显式添加 Write/Read Effect。
  
  if (hasPureTensorSemantics()) {
    // Tensor 模式下是 Pure 的，什么都不做，列表为空即可。
    return;
  }

  // 2. 如果操作的是 MemRef (Bufferization 之后)
  // Input 是读
  for (auto [index, operand] : llvm::enumerate(getDpsInputs())) {
    // 检查operand是否是MemRef类型
    if (!llvm::isa<MemRefType>(operand.getType())) continue;
    effects.emplace_back(MemoryEffects::Read::get(), 
                         &getOperation()->getOpOperand(index), /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
  
  // Output (Outs) 是写也是读
  for (OpOperand &operand : getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}
