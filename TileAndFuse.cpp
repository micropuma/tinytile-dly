#include "Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tutorial {

#define GEN_PASS_DEF_TUTORIALTILEANDFUSE
#include "Passes.h.inc"

class TutorialTileAndFuse final
    : public impl::TutorialTileAndFuseBase<TutorialTileAndFuse> {
  void runOnOperation() override {}
};

}  // namespace mlir::tutorial
