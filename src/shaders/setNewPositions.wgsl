////////////////////////////////////////////////////////////////////////////////
// Set New Positions
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
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  // This is just a simple compute shader to update the position
  // of agents based on their previously calculated velocity

  let idx = GlobalInvocationID.x;
  var agent = data.agents[idx];

  // Basic velocity integration
  agent.position = agent.position + sim_params.deltaTime * agent.velocity;

  // Store the new agent value
  data.agents[idx] = agent;
}
