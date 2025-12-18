#config = {
  parallel = [1, 1, 5, 64],
  reduction = [0, 0, 0, 0, 1, 1, 1]
}

!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

// NHWC × HWCF -> NHWC
module {
  func.func @conv(
      %input: !tinput,
      %filter: !tfilter,
      %bias: !tbias,
      %output: !toutput)  -> !toutput {
    %bias_init = tensor.empty() : !toutput
    %biased = linalg.broadcast ins(%bias : !tbias) outs(%bias_init : !toutput) dimensions = [0, 1, 2]

    %input_relu = tutorial.relu %input : !tinput , !tinput

    %convolved = linalg.generic {
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"],
      indexing_maps = [
        affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
      ]
    } 
    ins(%filter, %input_relu: !tfilter, !tinput) outs(%biased : !toutput)
    attrs = { lowering_config = #config } {    // a tiling interface with a lowering config
    ^bb0(%in: f32, %f: f32, %b: f32):
      %m1 = arith.mulf %in, %f  : f32
      %0 = arith.addf %b, %m1  : f32
      linalg.yield %0 : f32
    } -> !toutput

    %relued = tutorial.relu_dps ins(%convolved : !toutput) outs(%output : !toutput) -> !toutput
    return %relued : !toutput
  }
}
