#ifndef TUTORIAL_TD_
#define TUTORIAL_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

def Tutorial_Dialect : Dialect {
  let name = "tutorial";
  let cppNamespace = "::mlir::tutorial";
  let summary = [{
    Tutorial Dialect
  }];
  let description = [{
  }];
}

// Base class for Tutorial dialect ops that do not correspond to library calls.
class Tutorial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tutorial_Dialect, mnemonic, traits>;

// 主要起到演示目的
// 1. TilingInterface的使用
// 2. BufferizableOpInterface的使用
def Tutorial_DequantOp : Tutorial_Op<"dequant",
  [Pure,
   DeclareOpInterfaceMethods<BufferizableOpInterface, 
    ["bufferizesToMemoryRead",
     "getAliasingValues",  
     "bufferizesToMemoryWrite",
     "bufferize"]>,
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation",
     "generateResultTileValue",
     "getIterationDomainTileFromResultTile",
     "getIterationDomainTileFromOperandTile",
     "getTiledImplementationFromOperandTile"]>
  ]> {
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale
  );
  let results = (outs
    AnyRankedTensor:$output
  );
  let assemblyFormat = [{
    $input `,` $scale attr-dict `:` type($input) `,` type($scale) `,` type($output)
  }];
}

// 教学作用，简单的relu操作
def Tutorial_ReluOp : Tutorial_Op<"relu",
  [Pure,  
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
      "getLoopIteratorTypes",
      "getResultTilePosition",
      "getTiledImplementation",
      "generateResultTileValue",
      "getIterationDomainTileFromResultTile",
      "getIterationDomainTileFromOperandTile",
      "getTiledImplementationFromOperandTile"]>
  ]> {
  let arguments = (ins
    AnyRankedTensor:$input
  );
  let results = (outs
    AnyRankedTensor:$output
  );
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `,` type($output)
  }];
}

// 教学作用，DPS风格的relu操作
// 添加如下功能：  
// 1. DestinationStyleOpInterface接口
// 2. BufferizableOpInterface接口
// 3. verifier，确保input/output shape一致
// 4. folder，用来完成canonicalization
def Tutorial_ReluOpDPS : Tutorial_Op<"relu_dps",
  [Pure,  
   DestinationStyleOpInterface,   // DPS模式接口
   PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>,    // trait，确保input/output element type一致
   DeclareOpInterfaceMethods<TilingInterface,         // 由于是DPS模式，所以不需要generateResultTileValue等方法
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue",
       "getIterationDomainTileFromResultTile",
       "getIterationDomainTileFromOperandTile",
       "getTiledImplementationFromOperandTile"]>
  ]> {
  let summary = "Relu operation in destination passing style";
  let arguments = (ins
    AnyShaped:$input,
    AnyShaped:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let hasVerifier = 1;         // 启用verifier检查
}

#endif // TUTORIAL_TD_
