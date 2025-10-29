#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2)>
module {
  func.func @conv(%arg0: tensor<5x82x102x128xf32>, %arg1: tensor<128x3x3x128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<5x80x100x128xf32>) -> tensor<5x80x100x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<5x64xf32>
    %0 = bufferization.to_memref %arg0 : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : memref<128xf32, strided<[?], offset: ?>>
    %3 = bufferization.to_memref %arg3 : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x80x100x128xf32>
    memref.copy %3, %alloc : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<5x80x100x128xf32>
    scf.forall (%arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0) to (5, 80, 100, 128) step (1, 1, 5, 64) {
      %5 = vector.transfer_read %2[%arg7], %cst {in_bounds = [true]} : memref<128xf32, strided<[?], offset: ?>>, vector<64xf32>
      %6 = vector.broadcast %5 : vector<64xf32> to vector<5x64xf32>
      %7 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %6) -> (vector<5x64xf32>) {
        %9 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (vector<5x64xf32>) {
          %10 = scf.for %arg12 = %c0 to %c128 step %c1 iter_args(%arg13 = %arg11) -> (vector<5x64xf32>) {
            %11 = vector.transfer_read %1[%arg12, %arg8, %arg10, %arg7], %cst {in_bounds = [true]} : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<64xf32>
            %12 = vector.broadcast %11 : vector<64xf32> to vector<5x64xf32>
            %13 = affine.apply #map()[%arg5, %arg8]
            %14 = affine.apply #map()[%arg6, %arg10]
            %15 = vector.transfer_read %0[%arg4, %13, %14, %arg12], %cst {in_bounds = [true], permutation_map = #map1} : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<5xf32>
            %16 = vector.broadcast %15 : vector<5xf32> to vector<64x5xf32>
            %17 = vector.transpose %16, [1, 0] : vector<64x5xf32> to vector<5x64xf32>
            %18 = arith.mulf %12, %17 : vector<5x64xf32>
            %19 = arith.addf %arg13, %18 : vector<5x64xf32>
            scf.yield %19 : vector<5x64xf32>
          }
          scf.yield %10 : vector<5x64xf32>
        }
        scf.yield %9 : vector<5x64xf32>
      }
      %subview = memref.subview %alloc[%arg4, %arg5, %arg6, %arg7] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32> to memref<5x64xf32, strided<[128, 1], offset: ?>>
      %8 = arith.maxnumf %7, %cst_0 : vector<5x64xf32>
      vector.transfer_write %8, %alloc[%arg4, %arg5, %arg6, %arg7] {in_bounds = [true, true]} : vector<5x64xf32>, memref<5x80x100x128xf32>
      %subview_1 = memref.subview %alloc[%arg4, %arg5, %arg6, %arg7] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32> to memref<5x64xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview, %subview_1 : memref<5x64xf32, strided<[128, 1], offset: ?>> to memref<5x64xf32, strided<[128, 1], offset: ?>>
    }
    %4 = bufferization.to_tensor %alloc : memref<5x80x100x128xf32>
    return %4 : tensor<5x80x100x128xf32>
  }
}

