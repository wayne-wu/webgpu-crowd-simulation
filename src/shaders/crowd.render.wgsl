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
  cameraPos : vec3<f32>;
  agentScale : f32;
};

[[binding(0), group(0)]] var<uniform> render_params : RenderParams;

struct VertexInput {
  [[location(0)]] position : vec3<f32>;  // agent position (world space)
  [[location(1)]] color    : vec4<f32>;  // agent color
  [[location(2)]] velocity : vec3<f32>;  // agent velocity
  [[location(3)]] mesh_pos : vec4<f32>;  // mesh vertex position (model space)
  [[location(4)]] mesh_uv  : vec2<f32>;  // mesh vertex uv
  [[location(5)]] mesh_nor : vec4<f32>;  // mesh vertex normal
  [[location(6)]] mesh_col : vec3<f32>;  // mesh vertex color
};

struct VertexOutput {
  [[builtin(position)]] position : vec4<f32>;
  [[location(0)]]       color    : vec4<f32>;
  [[location(1)]]       mesh_pos : vec4<f32>;
  [[location(2)]]       mesh_uv  : vec2<f32>;
  [[location(3)]]       mesh_nor : vec4<f32>;
  [[location(4)]]       mesh_col : vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(in : VertexInput) -> VertexOutput {

  var vel = normalize(in.velocity);

  var model = mat4x4<f32>();
  var scale = mat4x4<f32>();
  scale[0] = vec4<f32>(render_params.agentScale, 0.0, 0.0, 0.0);
  scale[1] = vec4<f32>(0.0, render_params.agentScale, 0.0, 0.0);
  scale[2] = vec4<f32>(0.0, 0.0, render_params.agentScale, 0.0);
  scale[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var rot = mat4x4<f32>();
  var forward = vel;
  var up = vec3<f32>(0.0, 1.0, 0.0);
  var right = normalize(cross(forward.xyz, up.xyz));
  rot[0] = vec4<f32>(right, 0.0);
  rot[1] = vec4<f32>(up, 0.0);
  rot[2] = vec4<f32>(forward, 0.0);
  rot[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var trans = mat4x4<f32>();
  trans[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  trans[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  trans[2] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  trans[3] = vec4<f32>(in.position, 1.0);

  model = trans * rot * scale;

  var out : VertexOutput;  
  out.position = render_params.modelViewProjectionMatrix * model * in.mesh_pos;
  out.color = in.color;
  out.mesh_uv = in.mesh_uv;
  out.mesh_pos = in.mesh_pos;
  out.mesh_nor = rot * in.mesh_nor;
  out.mesh_col = in.mesh_col;
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
[[stage(fragment)]]
fn fs_main(in : VertexOutput) -> [[location(0)]] vec4<f32> {
  var cameraDir = in.position.xyz - render_params.cameraPos;
  var lightDir = vec4<f32>(1.0, 1.0, 1.0, 0.0);
  var lambertTerm = dot(normalize(lightDir), normalize(in.mesh_nor));

  var meshCol = vec4<f32>(in.mesh_col, 1.0);
  if (meshCol.r > 0.99 && meshCol.g > 0.99 && meshCol.b > 0.99){
    meshCol = in.color;
  }
  var albedo = in.color * 0.3 + meshCol * 0.7;
  return albedo + 0.5 * lambertTerm * vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
