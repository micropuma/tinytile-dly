#ifndef TUTORIAL_TD_
#define TUTORIAL_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

def Tutorial_Dialect : Dialect {
  let name = "tutorial";
  let cppNamespace = "::mlir::tutorial";
  let summary = [{
    Tutorial Dialect
  }];
  let description = [{
  }];
}

def Tutorial_DequantOp : Op<Tutorial_Dialect,
                              "dequant",
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

#endif // TUTORIAL_TD_
