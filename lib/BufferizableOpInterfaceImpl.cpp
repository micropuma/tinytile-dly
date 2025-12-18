#include "BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"

#include "mlir/IR/Operation.h"
#include "Tutorial.h"

using namespace mlir;
using namespace mlir::tutorial;
using namespace mlir::bufferization;

namespace {
struct ReluOpDPSInterface
    : public BufferizableOpInterface::ExternalModel<ReluOpDPSInterface,
                                                     tutorial::ReluOpDPS> {
  // In ReluOpDPS, the input operand is read, and the output operand is written.                                                    
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto reluOp = cast<tutorial::ReluOpDPS>(op);
    return &opOperand == &reluOp.getInputMutable();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 1;
  }

  // The output operand aliases with the result.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    if (opOperand.getOperandNumber() == 1) {
      return {{op->getOpResult(0), BufferRelation::Equivalent}};
    }
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto reluOp = cast<tutorial::ReluOpDPS>(op);
    // nice helper function from bufferization dialect to get the buffer for a value
    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, reluOp.getInput(), options);
    if (failed(inputBuffer))
      return failure();
    FailureOr<Value> outputBuffer =
        getBuffer(rewriter, reluOp.getOutput(), options);
    if (failed(outputBuffer))
      return failure();
    rewriter.create<tutorial::ReluOpDPS>(reluOp.getLoc(),
                                         /*result=*/TypeRange(), *inputBuffer,
                                         *outputBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *outputBuffer);
    return success();
  }
};
} // namespace

void mlir::tutorial::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TutorialDialect *dialect) {
    ReluOpDPS::attachInterface<ReluOpDPSInterface>(*ctx); 
  });
}