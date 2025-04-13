#ifndef TUTORIAL_COMPILER_PASSES_TD_
#define TUTORIAL_COMPILER_PASSES_TD_

include "mlir/Pass/PassBase.td"

def TutorialTileAndFuse : Pass<"tutorial-tile-and-fuse"> {
  let summary = "";
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect"];
}

#endif // TUTORIAL_COMPILER_PASSES_TD_
