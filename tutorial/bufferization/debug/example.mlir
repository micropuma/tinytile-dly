// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only print-conflicts"
func.func @test(%arg0: f32, %arg1: f32, %arg2: index, %arg3: index) -> (f32, tensor<3xf32>) {
  // Create a new tensor with [%arg0, %arg0, %arg0].
  %0 = tensor.from_elements %arg0, %arg0, %arg0 : tensor<3xf32>

  // Insert something into the new tensor.
  %1 = tensor.insert %arg1 into %0[%arg2] : tensor<3xf32>

  // Read from the old tensor.
  %r = tensor.extract %0[%arg3] : tensor<3xf32>

  // Return the extracted value and the result of the insertion.
  func.return %r, %1 : f32, tensor<3xf32>
}
