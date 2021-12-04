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
  [[location(2)]] velocity : vec4<f32>;  // agent velocity
  [[location(3)]] mesh_pos : vec4<f32>;  // mesh vertex position (model space)
  [[location(4)]] mesh_uv  : vec2<f32>;  // mesh vertex uv
};

struct VertexOutput {
  [[builtin(position)]] position : vec4<f32>;
  [[location(0)]]       color    : vec4<f32>;
  [[location(1)]]       mesh_pos : vec4<f32>;
  [[location(2)]]       mesh_uv  : vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(in : VertexInput) -> VertexOutput {

  var vel = normalize(in.velocity);
  //vel.y = 0.0;
  // TODO: How to construct mat4x4?
  var model = mat4x4<f32>();
  var scale = mat4x4<f32>();
  scale[0] = vec4<f32>(0.005, 0.0, 0.0, 0.0);
  scale[1] = vec4<f32>(0.0, 0.005, 0.0, 0.0);
  scale[2] = vec4<f32>(0.0, 0.0, 0.005, 0.0);
  scale[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var rot = mat4x4<f32>();
  var right = vel;
  var up = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  var forward = normalize(vec4<f32>(cross(right.xyz, up.xyz), 0.0));
  rot[0] = right;
  rot[1] = up;
  rot[2] = forward;
  rot[3] = scale[3];

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
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
[[stage(fragment)]]
fn fs_main(in : VertexOutput) -> [[location(0)]] vec4<f32> {
  //var lightPos = vec4<f32>(1, 1, 1);
  //var lambert = 
  return in.color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct SimulationParams {
  deltaTime : f32;
  seed : vec4<f32>;
};

struct Agent {
  position : vec3<f32>;
  lifetime : f32;
  color    : vec4<f32>;
  velocity : vec3<f32>;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> data : Agents;

[[stage(compute), workgroup_size(64)]]
fn simulate([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  rand_seed = (sim_params.seed.xy + vec2<f32>(GlobalInvocationID.xy)) * sim_params.seed.zw;

  let idx = GlobalInvocationID.x;
  var agent = data.agents[idx];

  // Basic velocity integration
  agent.position = agent.position + sim_params.deltaTime * agent.velocity;

  // Store the new agent value
  data.agents[idx] = agent;
}
