# TinyTile: A Naive TileAndFuse Tutorial

This repository contains code accompanied along with the tutorial ["An Introduction to Tensor Tiling in MLIR" given at EuroLLVM 2025](https://youtu.be/FLEb30WyroA?si=BBXuJofubQ_WpMIU).

This repository was forked from the tutorial source code, mainly for learning how to do basic tiling and fusion optimizations with the help of MLIR api. Some changes have been made in this repo:  
* More structured MLIR codes 
* Some useful MLIR components added for Debugging and practice   
    1. MLIR listener mechanism, refer to `include/DimListener.h`  
* A relu demo support tiling interface, and is in DPS manner, able to do producer fusion as well as consumer fusion.


The pipeline in tutorial support very basic 2D convolution tiling, i am also trying to make things more interesting: :fire:  
    * more advanced tiling approach following this [blog](https://www.lei.chat/posts/codegen-performant-convolution-kernels-for-mobile-gpus/).  
    * Minograd algorithm applied, following this slide [Winograd Convolutions in MLIR](http://www.harshmenon.ai/assets/MLIRSummit2022.Nodai.Menon.pdf).  
    * More advanced topics, to support decomposable operations in mlir, following this slide [Decomposable Operators in  IREE](http://www.harshmenon.ai/assets/C4ML2023.Nodai.Menon.pdf).