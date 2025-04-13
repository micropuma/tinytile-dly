#ifndef TUTORIAL_COMPILER_PASSES_H_
#define TUTORIAL_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::tutorial {

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}  // namespace mlir::tutorial

#endif  // TUTORIAL_COMPILER_PASSES_H_
