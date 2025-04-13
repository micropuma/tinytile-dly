#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tutorial {

#define GEN_PASS_DEF_TUTORIALVECTORIZATION
#include "Passes.h.inc"

namespace {

class TutorialVectorization final
    : public impl::TutorialVectorizationBase<TutorialVectorization> {
  using TutorialVectorizationBase::TutorialVectorizationBase;
  void runOnOperation() override;
};

}  // namespace

void TutorialVectorization::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp);

  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) { candidates.push_back(op); });

  for (linalg::GenericOp candidate : candidates) {
    (void)linalg::vectorize(rewriter, candidate);
  }

  // Cleanup.
  {
    RewritePatternSet vectorizationPatterns(funcOp.getContext());
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    vector::populateSinkVectorOpsPatterns(vectorizationPatterns);
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
    if (failed(
            applyPatternsGreedily(funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }
  }
}

}  // namespace mlir::tutorial
