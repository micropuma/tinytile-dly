# src/CMakeLists.txt

set(SRC_FILES
    TileAndFuse.cpp
    Vectorization.cpp
    ApplyTilingSpec.cpp
    Tutorial.cpp
    SliceListener.cpp
)

add_mlir_library(TutorialLib
    ${SRC_FILES}

    DEPENDS
    MLIRTutorialIncGen
    TutorialPassIncGen

    LINK_LIBS PUBLIC
    ${MLIR_DIALECT_LIBS}
    ${MLIR_CONVERSION_LIBS}
    MLIROptLib

    # 使用根目录下的find-name.sh快速定位符号所在的库
    MLIRIR
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRLinalgTransformOps   
    MLIRTensorTransformOps      
    MLIRSCFTransformOps    
    MLIRTensorInferTypeOpInterfaceImpl
)

target_include_directories(TutorialLib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)
