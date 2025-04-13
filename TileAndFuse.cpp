#include <queue>

#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tutorial {

#define GEN_PASS_DEF_TUTORIALTILEANDFUSE
#include "Passes.h.inc"

namespace {

class TutorialTileAndFuse final
    : public impl::TutorialTileAndFuseBase<TutorialTileAndFuse> {
  using TutorialTileAndFuseBase::TutorialTileAndFuseBase;
  void runOnOperation() override;
};

static FailureOr<TilingInterface> getLoweringConfigOp(func::FuncOp funcOp) {
  TilingInterface tilingOp;
  funcOp.walk([&](Operation* op) {
    if (auto tileOp = dyn_cast<TilingInterface>(op)) {
      if (tileOp->hasAttr("lowering_config")) {
        tilingOp = tileOp;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (tilingOp) {
    return tilingOp;
  }

  return failure();
}

static SmallVector<OpFoldResult> getTilingSizes(DictionaryAttr loweringConfig,
                                                tutorial::TilingLevel level) {
  switch (level) {
    case tutorial::TilingLevel::Parallel: {
      if (auto parallel = loweringConfig.getAs<ArrayAttr>("parallel")) {
        return llvm::map_to_vector(
            parallel.getAsRange<IntegerAttr>(),
            [](IntegerAttr x) { return OpFoldResult(x); });
      }
      break;
    }
    case tutorial::TilingLevel::Serial: {
      if (auto serial = loweringConfig.getAs<ArrayAttr>("serial")) {
        return llvm::map_to_vector(
            serial.getAsRange<IntegerAttr>(),
            [](IntegerAttr x) { return OpFoldResult(x); });
      }
      break;
    }
  }

  return SmallVector<OpFoldResult>{};
}

}  // namespace

void TutorialTileAndFuse::runOnOperation() {
  MLIRContext* context = &getContext();
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp);

  FailureOr<TilingInterface> maybeTilingOp = getLoweringConfigOp(funcOp);
  if (failed(maybeTilingOp)) {
    return;
  }

  TilingInterface tilingOp = maybeTilingOp.value();
  SmallVector<OpFoldResult> tileSizes = getTilingSizes(
      tilingOp->getAttrOfType<DictionaryAttr>("lowering_config"), tilingLevel);
  auto zero = rewriter.getIndexAttr(0);
  int64_t numLoops = tilingOp.getLoopIteratorTypes().size();
  tileSizes.resize(numLoops, zero);

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  if (tilingLevel == tutorial::TilingLevel::Parallel) {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  } else {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
  }

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  // Tile Operation and Fuse it's Producers.
  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp,
                                                tileAndFuseOptions);
  if (failed(tiledResults)) {
    return signalPassFailure();
  }

  for (Value result : tilingOp->getResults()) {
    rewriter.replaceAllUsesWith(result, tiledResults->replacements[result]);
  }

  // Fuse Consumers into the tiled operations.
  std::queue<Operation*> candidates;
  auto addCandidateSlices = [&candidates](Operation* fusedOp) {
    for (Operation* userOp : fusedOp->getResults().getUsers()) {
      if (isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(userOp)) {
        candidates.push(userOp);
      }
    }
  };

  for (Operation* tiledOp : tiledResults->tiledAndFusedOps) {
    addCandidateSlices(tiledOp);
  }

  MutableArrayRef<LoopLikeOpInterface> loops = tiledResults->loops;
  while (!candidates.empty()) {
    Operation* candidateSliceOp = candidates.front();
    candidates.pop();

    FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
        mlir::scf::tileAndFuseConsumerOfSlice(rewriter, candidateSliceOp,
                                              loops);
    if (failed(fusedResult)) {
      continue;
    }

    rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                       fusedResult->tiledOps.front());

    addCandidateSlices(fusedResult->tiledAndFusedConsumerOperand->getOwner());
  }

  // Cleanup.
  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace mlir::tutorial
