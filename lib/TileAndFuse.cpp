#include <queue>

#include "Passes.h"
#include "Tutorial.h"
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
    case tutorial::TilingLevel::Reduction: {
      if (auto reduction = loweringConfig.getAs<ArrayAttr>("reduction")) {
        return llvm::map_to_vector(
            reduction.getAsRange<IntegerAttr>(),
            [](IntegerAttr x) { return OpFoldResult(x); });
      }
      break;
    }
  }

  return SmallVector<OpFoldResult>{};
}

bool isDestinationSlice(tensor::ExtractSliceOp extractSliceOp) {
  auto blockArg = dyn_cast<BlockArgument>(extractSliceOp.getSource());
  if (blockArg && isa<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
    return true;
  }
  return false;
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

  struct SliceListener : public RewriterBase::Listener {
    void notifyOperationInserted(Operation* op,
                                 OpBuilder::InsertPoint) override {
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp,
              tensor::ParallelInsertSliceOp>(op)) {
        candidates.push_back(op);
      }
    }

    void notifyOperationReplaced(Operation* op, ValueRange) override {
      removeOp(op);
    };

    void notifyOperationErased(Operation* op) override { removeOp(op); };

    void removeOp(Operation* op) {
      auto it = llvm::find(candidates, op);
      if (it != candidates.end()) {
        candidates.erase(it);
      }
    }

    std::deque<Operation*> candidates;
  };

  SliceListener listener;
  rewriter.setListener(&listener);

  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingOp, tilingOptions);
  if (failed(tiledResults)) {
    return signalPassFailure();
  }
  rewriter.replaceOp(tilingOp, tiledResults->replacements);

  MutableArrayRef<LoopLikeOpInterface> loops = tiledResults->loops;
  std::deque<Operation*>& candidates = listener.candidates;
  while (!candidates.empty()) {
    Operation* candidate = candidates.front();
    candidates.pop_front();

    if (auto producerSlice = dyn_cast<tensor::ExtractSliceOp>(candidate)) {
      if (candidate->getUsers().empty()) {
        continue;
      }
      // Do not tile destination slices for reduction tiling.
      if (tilingLevel == tutorial::TilingLevel::Reduction &&
          isDestinationSlice(producerSlice)) {
        continue;
      }
      std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
          scf::tileAndFuseProducerOfSlice(rewriter, producerSlice, loops);
    }

    if (tilingLevel == tutorial::TilingLevel::Reduction) {
      // Do not do consumer fusion for reduction tiling.
      continue;
    }

    if (isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(candidate)) {
      FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
          scf::tileAndFuseConsumerOfSlice(rewriter, candidate);

      if (succeeded(fusedResult)) {
        rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                           fusedResult->tiledOps.front());
      }
    }
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
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace mlir::tutorial
