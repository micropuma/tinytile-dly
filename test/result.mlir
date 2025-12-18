#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2)>
module {
  func.func @conv(%arg0: memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg2: memref<128xf32, strided<[?], offset: ?>>, %arg3: memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> {
    %cst = arith.constant dense<0.000000e+00> : vector<5x64xf32>
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    scf.forall (%arg4, %arg5, %arg6, %arg7) = (0, 0, 0, 0) to (5, 80, 100, 128) step (1, 1, 5, 64) {
      %0 = vector.transfer_read %arg2[%arg7], %cst_0 {in_bounds = [true]} : memref<128xf32, strided<[?], offset: ?>>, vector<64xf32>
      %1 = vector.broadcast %0 : vector<64xf32> to vector<5x64xf32>
      %2 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %1) -> (vector<5x64xf32>) {
        %4 = scf.for %arg10 = %c0 to %c3 step %c1 iter_args(%arg11 = %arg9) -> (vector<5x64xf32>) {
          %5 = scf.for %arg12 = %c0 to %c128 step %c1 iter_args(%arg13 = %arg11) -> (vector<5x64xf32>) {
            %6 = vector.transfer_read %arg1[%arg12, %arg8, %arg10, %arg7], %cst_0 {in_bounds = [true]} : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<64xf32>
            %7 = vector.broadcast %6 : vector<64xf32> to vector<5x64xf32>
            %8 = affine.apply #map()[%arg5, %arg8]
            %9 = affine.apply #map()[%arg6, %arg10]
            %10 = vector.transfer_read %arg0[%arg4, %8, %9, %arg12], %cst_0 {in_bounds = [true], permutation_map = #map1} : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>, vector<5xf32>
            %11 = vector.broadcast %10 : vector<5xf32> to vector<64x5xf32>
            %12 = vector.transpose %11, [1, 0] : vector<64x5xf32> to vector<5x64xf32>
            %13 = arith.mulf %7, %12 : vector<5x64xf32>
            %14 = arith.addf %arg13, %13 : vector<5x64xf32>
            scf.yield %14 : vector<5x64xf32>
          }
          scf.yield %5 : vector<5x64xf32>
        }
        scf.yield %4 : vector<5x64xf32>
      }
      %subview = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
      %3 = arith.maxnumf %2, %cst : vector<5x64xf32>
      vector.transfer_write %3, %arg3[%arg4, %arg5, %arg6, %arg7] {in_bounds = [true, true]} : vector<5x64xf32>, memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
      %subview_1 = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7] [1, 1, 5, 64] [1, 1, 1, 1] : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
      memref.copy %subview, %subview_1 : memref<5x64xf32, strided<[?, ?], offset: ?>> to memref<5x64xf32, strided<[?, ?], offset: ?>>
    }
    return %arg3 : memref<5x80x100x128xf32, strided<[?, ?, ?, ?], offset: ?>>
  }
}

