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
