#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2)>
module {
  func.func @conv(%arg0: memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg2: memref<128xf32, strided<[?], offset: ?>>, %arg3: memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0) to (5, 80, 100, 128) step (1, 1, 5, 64) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x5x64xf32>
      %0 = vector.transfer_read %arg2[%arg7], %cst {in_bounds = [true]} : memref<128xf32, strided<[?], offset: ?>>, vector<64xf32>
      %1 = vector.broadcast %0 : vector<64xf32> to vector<5x64xf32>
      %2 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %1) -> (vector<5x64xf32>) {
        %7 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (vector<5x64xf32>) {
          %8 = scf.for %arg12 = %c0 to %c128 step %c1 iter_args(%arg13 = %arg11) -> (vector<5x64xf32>) {
            %9 = affine.apply #map()[%arg5, %arg8]
            %10 = affine.apply #map()[%arg6, %arg10]
            %subview_0 = memref.subview %arg0[%arg4, %9, %10, %arg12] [1, 1, 5, 1] [1, 1, 1, 1] : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %11 = bufferization.to_tensor %subview_0 : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %12 = tutorial.relu %11 : tensor<1x1x5x1xf32>, tensor<1x1x5x1xf32>
            %13 = bufferization.to_memref %12 : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %14 = vector.transfer_read %arg1[%arg12, %arg8, %arg10, %arg7], %cst {in_bounds = [true]} : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<64xf32>
            %15 = vector.broadcast %14 : vector<64xf32> to vector<5x64xf32>
            %16 = vector.transfer_read %13[%c0, %c0, %c0, %c0], %cst {in_bounds = [true], permutation_map = #map1} : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<5xf32>
            %17 = vector.broadcast %16 : vector<5xf32> to vector<64x5xf32>
            %18 = vector.transpose %17, [1, 0] : vector<64x5xf32> to vector<5x64xf32>
            %19 = arith.mulf %15, %18 : vector<5x64xf32>
            %20 = arith.addf %arg13, %19 : vector<5x64xf32>
            scf.yield %20 : vector<5x64xf32>
          }
          scf.yield %8 : vector<5x64xf32>
        }
        scf.yield %7 : vector<5x64xf32>
      }
      vector.transfer_write %2, %alloc[%c0, %c0, %c0, %c0] {in_bounds = [true, true]} : vector<5x64xf32>, memref<1x1x5x64xf32>
      %3 = bufferization.to_tensor %alloc : memref<1x1x5x64xf32>
      %subview = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x5x64xf32, strided<[?, ?, ?, ?], offset: ?>>
      %4 = bufferization.to_tensor %subview : memref<1x1x5x64xf32, strided<[?, ?, ?, ?], offset: ?>>
      %5 = tutorial.relu_dps ins(%3 : tensor<1x1x5x64xf32>) outs(%4 : tensor<1x1x5x64xf32>) -> tensor<1x1x5x64xf32>
      %6 = bufferization.to_memref %5 : memref<1x1x5x64xf32, strided<[?, ?, ?, ?], offset: ?>>
      memref.copy %6, %subview : memref<1x1x5x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x5x64xf32, strided<[?, ?, ?, ?], offset: ?>>
    }
    return %arg3 : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
  }
}

