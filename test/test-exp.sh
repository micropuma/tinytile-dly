tutorial-compiler \
    example-experiment.mlir \
    -debug-only=slice-listener \
    -debug-only=tile-and-fuse \
    -o result-tutorial.mlir \
    2>&1 | tee report-experiment.log