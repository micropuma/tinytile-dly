#ifndef TUTORIAL_H_
#define TUTORIAL_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

// Generated headers.
#include "TutorialDialect.h.inc"

#define GET_OP_CLASSES
#include "Tutorial.h.inc"

#endif  // TUTORIAL_H_
