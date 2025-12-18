#ifndef MLIR_DIALECT_TUTORIAL_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_TUTORIAL_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {

class DialectRegistry;

namespace tutorial {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace tutorial
} // namespace mlir

#endif // MLIR_DIALECT_TUTORIAL_BUFFERIZABLEOPINTERFACEIMPL_H