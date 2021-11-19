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

[[block]] struct PlannedPosData {
  positions : array<vec2<f32>>;
};

[[block]] struct GoalData {
  goals : array<vec3<f32>>;
};

[[block]] struct GridCells {
  cells : array<u32>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read_write> plannedPosData : PlannedPosData;
[[binding(3), group(0)]] var<storage, read_write> goalData : GoalData;
[[binding(4), group(0)]] var<storage, read_write> gridCell : GridCells;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // use calculated velocity to set the new position
  agent.position = agent.position + sim_params.deltaTime * agent.velocity;

  // Store the new agent value
  agentData.agents[idx] = agent;
}