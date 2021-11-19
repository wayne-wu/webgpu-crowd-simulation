[[block]] struct Matrix {
  size : vec2<f32>;
  numbers: array<f32>;
};

[[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
[[group(0), binding(1)]] var<storage, read> secondMatrix : Matrix;
[[group(0), binding(2)]] var<storage, write> resultMatrix : Matrix;

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
  // Guard against out-of-bounds work group sizes.
  if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(secondMatrix.size.y)) {
    return;
  }

  resultMatrix.size = vec2<f32>(firstMatrix.size.x, secondMatrix.size.y);
  
  let resultCell = vec2<u32>(global_id.x, global_id.y);
  var result = 0.0;
  for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
    let a = i + resultCell.x * u32(firstMatrix.size.y);
    let b = resultCell.y + i * u32(secondMatrix.size.y);
    result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
  }
  
  let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
  resultMatrix.numbers[index] = result;
}

[[stage(compute), workgroup_size(8)]]
fn doubleMat([[builtin(global_invocation_id)]] gidx : vec3<u32>) {

let idx = gidx.x; 
  resultMatrix.size = vec2<f32>(firstMatrix.size.x, secondMatrix.size.y);

  if (idx >= u32(resultMatrix.size.x) * u32(resultMatrix.size.y)){
    return;
  }

  resultMatrix.numbers[idx] = resultMatrix.numbers[idx] * 2.0;
}