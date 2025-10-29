#config = {
  parallel = [1, 1, 5, 64],
  reduction = [0, 0, 0, 0, 1, 1, 1]
}

!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

module {
  func.func @conv(
      %input: !tinput,
      %filter: !tfilter,
      %bias: !tbias,
      %output: !toutput)  -> !toutput {
    %bias_init = tensor.empty() : !toutput
    %biased = linalg.broadcast ins(%bias : !tbias)
      outs(%bias_init : !toutput) dimensions = [0, 1, 2]

    %convolved = linalg.generic {
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"],
      indexing_maps = [
        affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
      ]
    } 
    ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput)
    attrs = { lowering_config = #config } {
    ^bb0(%in: f32, %f: f32, %b: f32):
      %m1 = arith.mulf %in, %f  : f32
      %0 = arith.addf %b, %m1  : f32
      linalg.yield %0 : f32
    } -> !toutput

    %c0 = arith.constant 0.0 : f32
    %relued = linalg.generic {
      iterator_types = ["parallel", "parallel", "parallel", "parallel"],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> ()>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ]
    } ins(%c0, %convolved : f32, !toutput)
      outs(%output : !toutput) {
    ^bb0(%cst: f32, %in: f32, %out: f32):
      %0 = arith.maxnumf %cst, %in : f32
      linalg.yield %0 : f32
    } -> !toutput

    return %relued : !toutput
  }
}
