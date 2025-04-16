#include "Tutorial.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define GET_OP_CLASSES
#include "Tutorial.cpp.inc"

void mlir::tutorial::TutorialDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tutorial.cpp.inc"
      >();
}

#include "TutorialDialect.cpp.inc"

namespace mlir::tutorial {

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

FailureOr<TilingResult> DequantOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  int64_t rank = getInput().getType().getRank();
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
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

LogicalResult DequantOp::getIterationDomainTileFromOperandTile(
    OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
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

}  // namespace mlir::tutorial
