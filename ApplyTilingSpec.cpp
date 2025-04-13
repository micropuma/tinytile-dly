#include "Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tutorial {

#define GEN_PASS_DEF_TUTORIALAPPLYTILINGSPEC
#include "Passes.h.inc"

namespace {

class TutorialApplyTilingSpec final
    : public impl::TutorialApplyTilingSpecBase<TutorialApplyTilingSpec> {
  using TutorialApplyTilingSpecBase::TutorialApplyTilingSpecBase;
  void runOnOperation() override;
};

}  // namespace

void TutorialApplyTilingSpec::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  func::FuncOp funcOp = *moduleOp.getOps<func::FuncOp>().begin();

  auto entryPoint = funcOp->getAttrOfType<StringAttr>("transform_tiling_spec");
  if (!entryPoint) {
    return;
  }

  OpPassManager modulePassManager(ModuleOp::getOperationName());
  transform::InterpreterPassOptions options;
  options.entryPoint = entryPoint.str();
  modulePassManager.addPass(transform::createInterpreterPass(options));

  if (failed(runPipeline(modulePassManager, moduleOp))) {
    moduleOp.emitOpError("failed to run transform dialect passes");
    return signalPassFailure();
  }
}

}  // namespace mlir::tutorial
