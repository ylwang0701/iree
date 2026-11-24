// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// Scaled input maps cannot be vectorized before distribution.
// CHECK-LABEL: func.func @reduction_scaled_input
// CHECK-NOT: pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK: return
func.func @reduction_scaled_input(
    %input: tensor<32x256xf32>, %init: tensor<16xf32>,
    %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>) {
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m * 2, k)>, affine_map<(m, k) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<32x256xf32>) outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

// Check the producer even when the reduction has projected maps.
// CHECK-LABEL: func.func @reduction_scaled_producer
// CHECK-NOT: pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK: return
func.func @reduction_scaled_producer(
    %input: tensor<32x256xf32>, %init: tensor<16xf32>,
    %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>) {
  %empty = tensor.empty() : tensor<16x256xf32>
  %square = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m * 2, k)>, affine_map<(m, k) -> (m, k)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<32x256xf32>) outs(%empty : tensor<16x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sq = arith.mulf %in, %in : f32
    linalg.yield %sq : f32
  } -> tensor<16x256xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m, k)>, affine_map<(m, k) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%square : tensor<16x256xf32>) outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

// Literal zeros in the fused producer's output and consumer's input are allowed.
// CHECK-LABEL: func.func @reduction_broadcast_producer
// CHECK-SAME: pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK: %[[SQUARE:.+]] = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>]
// CHECK: linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0)>]
// CHECK-SAME: ins({{.*}}, %[[SQUARE]] :
func.func @reduction_broadcast_producer(
    %input: tensor<16x256xf32>, %bias: tensor<1x256xf32>, %init: tensor<16xf32>,
    %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>) {
  %empty = tensor.empty() : tensor<1x256xf32>
  %square = linalg.generic {
      indexing_maps = [affine_map<(u, k) -> (u, k)>, affine_map<(u, k) -> (0, k)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%bias : tensor<1x256xf32>) outs(%empty : tensor<1x256xf32>) {
  ^bb0(%b: f32, %out: f32):
    %sq = arith.mulf %b, %b : f32
    linalg.yield %sq : f32
  } -> tensor<1x256xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, k) -> (m, k)>, affine_map<(m, k) -> (0, k)>, affine_map<(m, k) -> (m)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input, %square : tensor<16x256xf32>, tensor<1x256xf32>) outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %val = arith.addf %in, %b : f32
    %sum = arith.addf %val, %out : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}
