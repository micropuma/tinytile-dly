# TinyTile: A TileAndFuse Tutorial

This repository contains code accompanied along with the tutorial ["An Introduction to Tensor Tiling in MLIR" given at EuroLLVM 2025](https://youtu.be/FLEb30WyroA?si=BBXuJofubQ_WpMIU).

This repository was forked from the tutorial source code, but adds the following improvements:   
* More structured MLIR codes
* Some useful MLIR components added for Debugging and practice   
    1. MLIR listener mechanism, refer to `include/DimListener.h`
* The pipeline in tutorial support very basic 2D convolution tiling, i am goint try to make things more interesting:  
    * more advanced approach following this [blog](https://www.lei.chat/posts/codegen-performant-convolution-kernels-for-mobile-gpus/).  
    * Minograd algorithm applied