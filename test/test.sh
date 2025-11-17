tutorial-compiler \
    example.mlir \
    -debug-only=slice-listener \
    -o result.mlir \
    2>&1 | tee report.log
