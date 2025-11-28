#define DEBUG_TYPE "slice-listener"

#include "SliceListener.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <algorithm>

using namespace mlir;

void SliceListener::notifyOperationInserted(
    Operation* op, OpBuilder::InsertPoint) {
  if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
          tensor::ParallelInsertSliceOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inserted: " << *op << "\n");
    candidates.push_back(op);
  }
}

void SliceListener::notifyOperationReplaced(Operation* op, ValueRange) {
  LLVM_DEBUG(llvm::dbgs() << "Replaced: " << *op << "\n");
  removeOp(op);
}

void SliceListener::notifyOperationErased(Operation* op) {
  LLVM_DEBUG(llvm::dbgs() << "Erased: " << *op << "\n");
  removeOp(op);
}

void SliceListener::removeOp(Operation* op) {
  auto it = llvm::find(candidates, op);
  if (it != candidates.end())
    candidates.erase(it);
}
