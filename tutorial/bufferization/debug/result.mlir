module {
  func.func @test(%arg0: f32, %arg1: f32, %arg2: index, %arg3: index) -> (f32, memref<3xf32>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3xf32>
    memref.store %arg0, %alloc[%c0] : memref<3xf32>
    memref.store %arg0, %alloc[%c1] : memref<3xf32>
    memref.store %arg0, %alloc[%c2] : memref<3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<3xf32>
    memref.copy %alloc, %alloc_0 : memref<3xf32> to memref<3xf32>
    memref.store %arg1, %alloc_0[%arg2] : memref<3xf32>
    %0 = memref.load %alloc[%arg3] : memref<3xf32>
    return %0, %alloc_0 : f32, memref<3xf32>
  }
}

