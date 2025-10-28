# include/CMakeLists.txt

# 生成 MLIR Dialect
add_mlir_dialect(Tutorial tutorial)

# 生成 Pass 声明
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tutorial)
add_public_tablegen_target(TutorialPassIncGen)

# 确保生成的 inc 文件可以被外部 include 到
include_directories(${CMAKE_CURRENT_BINARY_DIR})
