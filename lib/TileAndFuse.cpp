#include <queue>

#include "Passes.h"
#include "Tutorial.h"
#include "SliceListener.h"
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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tile-and-fuse"

namespace mlir::tutorial {

#define GEN_PASS_DEF_TUTORIALTILEANDFUSE
#include "Passes.h.inc"

namespace {

class TutorialTileAndFuse final
    : public impl::TutorialTileAndFuseBase<TutorialTileAndFuse> {
  using TutorialTileAndFuseBase::TutorialTileAndFuseBase;
  void runOnOperation() override;
};

// 获取funcOp的rooting tiling op（带有lowering_config属性的op）
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
  LLVM_DEBUG(llvm::dbgs() << "Tiling operation: " << tilingOp << "\n";);
  SmallVector<OpFoldResult> tileSizes = getTilingSizes(
      tilingOp->getAttrOfType<DictionaryAttr>("lowering_config"), tilingLevel);
  auto zero = rewriter.getIndexAttr(0);
  int64_t numLoops = tilingOp.getLoopIteratorTypes().size();
  tileSizes.resize(numLoops, zero);

  // // =================================================================
  // // [新增] Reduction Splitting 逻辑
  // // =================================================================
  // if (tilingLevel == tutorial::TilingLevel::Reduction) {
  //   // 假设我们在 lowering_config 里藏了一个 split_factor
  //   // 或者这里暂时硬编码测试，例如 split factor = 4
  //   int64_t splitFactor = 4; 
    
  //   // 配置 Split 选项
  //   mlir::linalg::SplitReductionOptions splitOptions;
  //   splitOptions.ratio = splitFactor; 
  //   splitOptions.index = tilingOp.getLoopIteratorTypes().size() - 1; 
  //   splitOptions.innerParallel = true; // 让拆分出来的维度变成并行的 (Parallel Reduction)

  //   // 执行 Split Reduction
  //   // 注意：splitReduction 只对 LinalgOp 有效，所以我们要 cast 一下
  //   if (auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
        
  //     LLVM_DEBUG(llvm::dbgs() << "Attempting split reduction on: " << *linalgOp << "\n");
      
  //     FailureOr<linalg::SplitReductionResult> splitResult = 
  //         mlir::linalg::splitReduction(rewriter, linalgOp,
  //             [&](OpBuilder &b, Operation *op, Value input) -> mlir::linalg::SplitReductionOptions {
  //                 return splitOptions;
  //             });

  //     if (succeeded(splitResult)) {
  //       // 关键点：Split 后，原始 op 被替换了。
  //       // splitResult->splitLinalgOp 是拆分后的核心计算 Op
  //       // splitResult->mergeOp 是后续的归约合并 Op
        
  //       // 我们通常希望对 *拆分后* 的 Op 进行 Tiling
  //       if (auto newTilingOp = dyn_cast<TilingInterface>(splitResult->splitLinalgOp.getOperation())) {
  //           tilingOp = newTilingOp;
  //           LLVM_DEBUG(llvm::dbgs() << "Split success. New tiling target: " << tilingOp << "\n");
  //       }
  //     }
  //   }
  // }
  // // =================================================================

  // LLVM_DEBUG(llvm::dbgs() << "Tiling operation: " << tilingOp << "\n";);

  // use scftiling options to help tiling
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  if (tilingLevel == tutorial::TilingLevel::Parallel) {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  } else {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
  }

  SliceListener listener;
  rewriter.setListener(&listener);

  // 实际的tiling 执行
  // 关注：（1）tileResults返回值（2）tileUsingSCF的具体实现
  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingOp, tilingOptions);
  if (failed(tiledResults)) {
    return signalPassFailure();
  }
  rewriter.replaceOp(tilingOp, tiledResults->replacements);

  LLVM_DEBUG(llvm::dbgs() << "Current FuncOp is: " << funcOp << "\n";);

  // do fusing producer/consumer into tile loops
  MutableArrayRef<LoopLikeOpInterface> loops = tiledResults->loops;
  std::deque<Operation*>& candidates = listener.candidates;

  LLVM_DEBUG(llvm::dbgs() << "Number of candidates for fusion: " << candidates.size() << "\n";);
  for (const auto& candidate : candidates) {
    LLVM_DEBUG(llvm::dbgs() << "Candidate for fusion: " << *candidate << "\n";);
  }

  // 尝试fusing producer/consumer into tile loops
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

      LLVM_DEBUG(llvm::dbgs() << "candidate is: " << *candidate << "\n";);

      // 尝试将producer fuse进tile loops
      std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
          scf::tileAndFuseProducerOfSlice(rewriter, producerSlice, loops);
      LLVM_DEBUG(llvm::dbgs() << "After producer fusion " << funcOp << "\n";);
    }

    if (tilingLevel == tutorial::TilingLevel::Reduction) {
      // Do not do consumer fusion for reduction tiling.
      continue;
    }

    // 将consumer fuse进tile loops
    if (isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(candidate)) {
      LLVM_DEBUG(llvm::dbgs() << "candidate is: " << *candidate << "\n";);
      FailureOr<scf::SCFFuseConsumerOfSliceResult> fusedResult =
          scf::tileAndFuseConsumerOfSlice(rewriter, candidate);

      if (succeeded(fusedResult)) {
        rewriter.replaceOp(fusedResult->origConsumerOperand->getOwner(),
                           fusedResult->tiledOps.front());
        LLVM_DEBUG(llvm::dbgs() << "After consumer fusion " << funcOp << "\n";);
      }
    }
  }

  // Cleanup.
  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  // 处理<1x2x2>变成<2x2>这种rank-reduced的情况
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
