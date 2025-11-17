#!/bin/bash

# === 用户配置区域 ===
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"
LLVM_INSTALL_DIR="/home/douliyang/large/mlir-workspace/tinytile-dly/third_party/llvm/install"

# === 检查 LLVM 目录是否存在 ===
if [ ! -d "$LLVM_INSTALL_DIR" ]; then
    echo "❌ 错误: LLVM 安装目录不存在: $LLVM_INSTALL_DIR"
    echo "请确认你已经执行了 ninja install 或修改此路径。"
    exit 1
fi

MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"

if [ ! -f "$MLIR_DIR/MLIRConfig.cmake" ]; then
    echo "❌ 错误: 找不到 MLIRConfig.cmake 于 $MLIR_DIR"
    exit 1
fi

if [ ! -f "$LLVM_DIR/LLVMConfig.cmake" ]; then
    echo "❌ 错误: 找不到 LLVMConfig.cmake 于 $LLVM_DIR"
    exit 1
fi

# === 处理 build 目录 ===
if [ -d "$BUILD_DIR" ]; then
    echo "⚠️ 检测到已有 build 目录，正在删除以执行干净构建..."
    rm -rf "$BUILD_DIR"
fi

echo "📁 创建新构建目录: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# === 运行 CMake 配置 ===
echo "⚙️ 运行 CMake..."
cmake .. \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON

# === 编译 ===
echo "🚀 开始编译..."
make -j"$(nproc)"

# === 构建完成 ===
if [ $? -eq 0 ]; then
    echo "✅ 构建完成！产物位于: $BUILD_DIR"
else
    echo "❌ 构建失败！"
fi
