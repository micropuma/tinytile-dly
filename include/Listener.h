#ifndef SCF_LISTENER_H_
#define SCF_LISTENER_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <deque>
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "slice-listener"

namespace mlir {

class Operation;

/// A rewriter that keeps track of all scf::ScfOp.
// 监控tiling的过程中产生的slice操作，以便后续进行融合。
struct SliceListener : public RewriterBase::Listener {
  void notifyOperationInserted(Operation* op,
                                OpBuilder::InsertPoint) override {
    if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
            tensor::ParallelInsertSliceOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "notifyOperationInserted: " << *op << "\n");
      candidates.push_back(op);
    }
  }

  void notifyOperationReplaced(Operation* op, ValueRange) override {
    LLVM_DEBUG(llvm::dbgs() << "notifyOperationReplaced: " << *op << "\n");
    removeOp(op);
  };

  void notifyOperationErased(Operation* op) override { 
    LLVM_DEBUG(llvm::dbgs() << "notifyOperationErased: " << *op << "\n");
    removeOp(op); 
  };

  void removeOp(Operation* op) {
    auto it = llvm::find(candidates, op);
    if (it != candidates.end()) {
      candidates.erase(it);
    }
  }

  std::deque<Operation*> candidates;
};

}  // namespace mlir

#endif // SCF_LISTENER_H_
