#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2)>
module {
  func.func @conv_with_dequant(%arg0: memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg2: memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg3: memref<128xf32, strided<[?], offset: ?>>, %arg4: memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> {
    %cst = arith.constant dense<0.000000e+00> : vector<5x64xf32>
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    scf.forall (%arg5, %arg6, %arg7, %arg8) = (0, 0, 0, 0) to (5, 80, 100, 128) step (1, 1, 5, 64) {
      %0 = vector.transfer_read %arg3[%arg8], %cst_0 {in_bounds = [true]} : memref<128xf32, strided<[?], offset: ?>>, vector<64xf32>
      %1 = vector.broadcast %0 : vector<64xf32> to vector<5x64xf32>
      %2 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %1) -> (vector<5x64xf32>) {
        %4 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %arg10) -> (vector<5x64xf32>) {
          %5 = scf.for %arg13 = %c0 to %c128 step %c1 iter_args(%arg14 = %arg12) -> (vector<5x64xf32>) {
            %6 = affine.apply #map()[%arg6, %arg9]
            %7 = affine.apply #map()[%arg7, %arg11]
            %subview_2 = memref.subview %arg0[%arg5, %6, %7, %arg13] [1, 1, 5, 1] [1, 1, 1, 1] : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %8 = bufferization.to_tensor %subview_2 : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %9 = affine.apply #map()[%arg6, %arg9]
            %10 = affine.apply #map()[%arg7, %arg11]
            %subview_3 = memref.subview %arg1[%arg5, %9, %10, %arg13] [1, 1, 5, 1] [1, 1, 1, 1] : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %11 = bufferization.to_tensor %subview_3 : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %12 = tutorial.dequant %8, %11 : tensor<1x1x5x1xf32>, tensor<1x1x5x1xf32>, tensor<1x1x5x1xf32>
            %13 = bufferization.to_memref %12 : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>
            %14 = vector.transfer_read %arg2[%arg13, %arg9, %arg11, %arg8], %cst_0 {in_bounds = [true]} : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<64xf32>
            %15 = vector.broadcast %14 : vector<64xf32> to vector<5x64xf32>
            %16 = vector.transfer_read %13[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true], permutation_map = #map1} : memref<1x1x5x1xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<5xf32>
            %17 = vector.broadcast %16 : vector<5xf32> to vector<64x5xf32>
            %18 = vector.transpose %17, [1, 0] : vector<64x5xf32> to vector<5x64xf32>
            %19 = arith.mulf %15, %18 : vector<5x64xf32>
            %20 = arith.addf %arg14, %19 : vector<5x64xf32>
            scf.yield %20 : vector<5x64xf32>
          }
          scf.yield %5 : vector<5x64xf32>
        }
        scf.yield %4 : vector<5x64xf32>
      }
      %subview = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
      %3 = arith.maxnumf %2, %cst : vector<5x64xf32>
      vector.transfer_write %3, %arg4[%arg5, %arg6, %arg7, %arg8] {in_bounds = [true, true]} : vector<5x64xf32>, memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
      %subview_1 = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
      memref.copy %subview, %subview_1 : memref<5x64xf32, strided<[?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
    }
    return %arg4 : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
  }
}

