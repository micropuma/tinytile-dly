// Remove the "transform_tiling_spec" attribute and uncomment the lowering
// config to try out the pass pipeline tiling
#config = {
  parallel = [1, 1, 5, 64],
  reduction = [0, 0, 0, 0, 1, 1, 1]
}

!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

module attributes { transform.with_named_sequence } {
  func.func @conv(
      %input: !tinput,
      %filter: !tfilter,
      %bias: !tbias,
      %output: !toutput)  -> !toutput
    attributes { }
  {
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
      %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
      %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
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
      %0 = arith.maxnumf %cst, %in {fastmath = #arith.fastmath<fast>} : f32
      linalg.yield %0 : f32
    } -> !toutput

    return %relued : !toutput
  }

  transform.named_sequence @__halide(
      %arg0: !transform.any_op) {

    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %conv, %relu = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %relu2, %co = transform.structured.tile_using_forall %relu
                                                        tile_sizes [0, 0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2
                                                        tile_sizes [1, 1, 5, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
      into %n_y_xo
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    %bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2
      into %n_y_xo2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op

    %red_fill, %conv4, %combining, %rz_ry_rx
    = transform.structured.tile_reduction_using_for %conv3 by
      tile_sizes=[0, 0, 0, 0, 1, 1, 1]
      : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)

    transform.yield
  }
}
