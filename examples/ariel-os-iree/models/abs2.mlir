func.func @main(%input : tensor<2xf32>) -> (tensor<2xf32>) {
  %result = math.absf %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
