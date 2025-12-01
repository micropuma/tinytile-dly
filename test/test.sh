# a transform dialect pipeline
tutorial-compiler \
    example-transform.mlir \
    -debug-only=slice-listener \
    -debug-only=tile-and-fuse \
    -o result.mlir \
    2>&1 | tee report-transform.log

# a convolution tiling pipeline
tutorial-compiler \
    example.mlir \
    -debug-only=slice-listener \
    -debug-only=tile-and-fuse \
    -o result.mlir \
    2>&1 | tee report.log

# experimental pipeline  
# the tutorial dequant op supports tiling  
# but don't support vectorization and bufferization yet  
tutorial-compiler \
    example-tutorial.mlir \
    2>&1 | tee report-experiment.log
