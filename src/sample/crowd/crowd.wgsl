////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////
var<private> rand_seed : vec2<f32>;

fn rand() -> f32 {
    rand_seed.x = fract(cos(dot(rand_seed, vec2<f32>(23.14077926, 232.61690225))) * 136.8168);
    rand_seed.y = fract(cos(dot(rand_seed, vec2<f32>(54.47856553, 345.84153136))) * 534.7645);
    return rand_seed.y;
}

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct RenderParams {
  modelViewProjectionMatrix : mat4x4<f32>;
};
[[binding(0), group(0)]] var<uniform> render_params : RenderParams;

struct VertexInput {
  [[location(0)]] position : vec3<f32>;  // agent position (world space)
  [[location(1)]] color    : vec4<f32>;  // agent color
  [[location(2)]] mesh_pos : vec4<f32>;  // mesh vertex position (model space)
  [[location(3)]] mesh_uv  : vec2<f32>;  // mesh vertex uv
};

struct VertexOutput {
  [[builtin(position)]] position : vec4<f32>;
  [[location(0)]]       color    : vec4<f32>;
  [[location(1)]]       mesh_pos : vec4<f32>;
  [[location(2)]]       mesh_uv  : vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(in : VertexInput) -> VertexOutput {

  // TODO: How to construct mat4x4?
  var model = mat4x4<f32>();
  model[0] = vec4<f32>(0.1, 0.0, 0.0, 0.0);
  model[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  model[2] = vec4<f32>(0.0, 0.0, 0.1, 0.0);
  model[3] = vec4<f32>(in.position, 1.0);

  var out : VertexOutput;  
  out.position = render_params.modelViewProjectionMatrix * model * in.mesh_pos;
  out.color = in.color;
  out.mesh_uv = in.mesh_uv;
  out.mesh_pos = in.mesh_pos;
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
[[stage(fragment)]]
fn fs_main(in : VertexOutput) -> [[location(0)]] vec4<f32> {
  return in.color;
}
