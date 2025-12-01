# dump the bufferization analysis
mlir-opt ./example.mlir \
         -one-shot-bufferize="bufferize-function-boundaries test-analysis-only print-conflicts" \
         2>&1 | tee bufferize.log

mlir-opt ./example.mlir \
    -one-shot-bufferize="bufferize-function-boundaries allow-unknown-ops" \
    -canonicalize \
    -cse \
    -fold-memref-alias-ops \
    -o result.mlir

# the whole bufferization pipeline
