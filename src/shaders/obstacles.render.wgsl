////////////////////////////////////////////////////////////////////////////////
// Obstacles Rendering
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct RenderParams {
  modelViewProjectionMatrix : mat4x4<f32>;
};
[[binding(0), group(0)]] var<uniform> render_params : RenderParams;

struct VertexInput {
  [[location(0)]] position : vec3<f32>;  // obstacle position (world space)
  [[location(1)]] rotation : f32;
  [[location(2)]] scale    : vec3<f32>;
  [[location(3)]] mesh_pos : vec4<f32>;  // mesh vertex position (model space)
  [[location(4)]] mesh_uv  : vec2<f32>;  // mesh vertex uv
};

struct VertexOutput {
  [[builtin(position)]] position : vec4<f32>;
  [[location(0)]]       color    : vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(in : VertexInput) -> VertexOutput {

  let c = cos(in.rotation);
  let s = sin(in.rotation);
  var model = mat4x4<f32>();
  model[0] = vec4<f32>(in.scale.x*c, 0.0, -s, 0.0);
  model[1] = vec4<f32>(0.0, in.scale.y, 0.0, 0.0);
  model[2] = vec4<f32>(s, 0.0, in.scale.z*c, 0.0);
  model[3] = vec4<f32>(in.position, 1.0);

  var out : VertexOutput;  
  out.position = render_params.modelViewProjectionMatrix * model * in.mesh_pos;
  out.color = vec4<f32>(0.1);
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
[[stage(fragment)]]
fn fs_main(in : VertexOutput) -> [[location(0)]] vec4<f32> {
  return in.color;
}
