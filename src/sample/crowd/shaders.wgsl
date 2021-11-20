//////////////////////////////////////////////////////////////////////
//                General Vertex Shader                             //
////////////////////////////////////////////////////////////////////// 

[[block]] struct Uniforms {
    modelViewProjectionMatrix : mat4x4<f32>;
  };
  [[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

[[block]] struct Neighbors {
  neighbors : array<u32>;
};
[[binding(1), group(0)]] var<storage> neighborData : Neighbors;
  
  struct VertexOutput {
    [[builtin(position)]] Position : vec4<f32>;
    [[location(0)]] fragUV : vec2<f32>;
    [[location(1)]] fragPosition: vec4<f32>;
  };
  
  [[stage(vertex)]]
  fn vs_main([[location(0)]] position : vec4<f32>,
          [[location(1)]] uv : vec2<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix * position;
    output.fragUV = uv;
    output.fragPosition = position;
    return output;
  }

//////////////////////////////////////////////////////////////////////
//            Fragment Shader for GridLines                         //
////////////////////////////////////////////////////////////////////// 

[[stage(fragment)]]
fn fs_gridLines([[location(0)]] fragUV: vec2<f32>,
        [[location(1)]] fragPosition: vec4<f32>) -> [[location(0)]] vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

//////////////////////////////////////////////////////////////////////
//            Fragment Shader for Platform                          //
////////////////////////////////////////////////////////////////////// 

[[stage(fragment)]]
fn fs_platform([[location(0)]] fragUV: vec2<f32>,
        [[location(1)]] fragPosition: vec4<f32>) -> [[location(0)]] vec4<f32> {
  return vec4<f32>(0.5, 0.5, 0.5, 1.0);
}
