tutorial-compiler \
    example.mlir \
    -debug-only=slice-listener \
    -debug-only=tile-and-fuse \
    -o result.mlir \
    2>&1 | tee report.log
