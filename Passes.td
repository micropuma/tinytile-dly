#ifndef TUTORIAL_COMPILER_PASSES_TD_
#define TUTORIAL_COMPILER_PASSES_TD_

include "mlir/Pass/PassBase.td"

def TutorialTileAndFuse : Pass<"tutorial-tile-and-fuse", "func::FuncOp"> {
  let summary = "";
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect"];
  let options = [
    Option<"tilingLevel", "tiling-level", "tutorial::TilingLevel",
    "/*default=*/tutorial::TilingLevel::Parallel",
    "",
    [{llvm::cl::values(
      clEnumValN(tutorial::TilingLevel::Parallel, "parallel", "scf.forall"),
      clEnumValN(tutorial::TilingLevel::Reduction, "Reduction", "scf.for")
    )}]>
  ];
}

def TutorialVectorization : Pass<"tutorial-vectorization", "func::FuncOp"> {
  let summary = "";
  let dependentDialects = ["linalg::LinalgDialect", "vector::VectorDialect"];
}

def TutorialApplyTilingSpec : Pass<"tutorial-apply-tiling-spec", "ModuleOp"> {
  let summary = "";
  let dependentDialects = ["transform::TransformDialect"];
}

#endif // TUTORIAL_COMPILER_PASSES_TD_
