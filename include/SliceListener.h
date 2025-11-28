#ifndef SCLICE_LISTENER_H_
#define SCLICE_LISTENER_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <deque>

namespace mlir {

class Operation;

/// Listener that records slice ops created during tiling.
struct SliceListener : public RewriterBase::Listener {
  void notifyOperationInserted(Operation* op,
                               OpBuilder::InsertPoint) override;

  void notifyOperationReplaced(Operation* op, ValueRange) override;

  void notifyOperationErased(Operation* op) override;

  void removeOp(Operation* op);

  std::deque<Operation*> candidates;
};

}  // namespace mlir

#endif // SCLICE_LISTENER_H_
