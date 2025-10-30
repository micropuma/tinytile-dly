#!/bin/bash

# LLVM 自动化构建 + 安装脚本

# 1. 基本配置变量
BUILD_DIR="./build"               # 构建目录
SRC_DIR="../llvm"                 # LLVM 源码路径
INSTALL_DIR="../install"          # ✅ 安装目录
BUILD_TYPE="Debug"                # Debug / Release
ENABLE_ASSERTIONS="ON"
ENABLE_PROJECTS="mlir;llvm;lld"
TARGETS_TO_BUILD="host"

echo "=== 开始配置并构建 LLVM ==="

# 2. 创建并进入构建目录
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    echo "创建构建目录: $BUILD_DIR"
fi
cd "$BUILD_DIR"

# 3. 检查源码目录
if [ ! -d "$SRC_DIR" ]; then
    echo "❌ 错误: 找不到 LLVM 源码目录 '$SRC_DIR'"
fi

# 4. CMake 配置阶段
echo "阶段 1: 运行 CMake 配置..."
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLLVM_ENABLE_ASSERTIONS=$ENABLE_ASSERTIONS \
    -DLLVM_ENABLE_PROJECTS=$ENABLE_PROJECTS \
    -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \         # ✅ 指定安装目录
    $SRC_DIR

if [ $? -ne 0 ]; then
    echo "❌ CMake 配置失败"
fi
echo "✅ CMake 配置成功"

# 5. 编译阶段
NUM_JOBS=$(nproc)
echo "阶段 2: 使用 Ninja 编译 ($NUM_JOBS 并行任务)..."
ninja -j $NUM_JOBS
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
fi
echo "✅ 编译成功"

# 6. 安装阶段
echo "阶段 3: 安装到 $INSTALL_DIR ..."
ninja install
if [ $? -ne 0 ]; then
    echo "❌ 安装失败"
fi

echo "🎉 LLVM + MLIR 安装成功到: $INSTALL_DIR"
