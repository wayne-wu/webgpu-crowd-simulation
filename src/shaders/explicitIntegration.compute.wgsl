////////////////////////////////////////////////////////////////////////////////
// Explicit Integration for Advecting the Agents
////////////////////////////////////////////////////////////////////////////////

let ksi : f32 = 0.0385;  // paper = 0.0385
let preferredVelocity : f32 = 1.4; // paper = 1.4

let maxNeighbors : u32 = 20u;

[[block]] struct SimulationParams {
  deltaTime : f32;
  avoidance : i32;
  seed : vec4<f32>;
};

struct Agent {
  x  : vec3<f32>;  // position + radius
  r  : f32;
  c  : vec4<f32>;  // color
  v  : vec3<f32>;  // velocity + inverse mass
  w  : f32;
  xp : vec3<f32>;  // planned/predicted position
  goal : vec3<f32>;
  nearNeighbors : array<u32, 20>; 
  farNeighbors : array<u32, 20>;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

fn getVelocityFromPlanner(agent : Agent) -> vec3<f32> {
  // TODO: Implement a more complex planner
  return normalize(agent.goal - agent.x) * preferredVelocity;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // velcity planning
  var vp = getVelocityFromPlanner(agent);

  // 4.1 Velocity Blending
  agent.v = (1.0 - ksi) * agent.v + ksi * vp;

  // explicit integration
  agent.xp = agent.x + sim_params.deltaTime * agent.v;

  // Store the new agent value
  agentData.agents[idx] = agent;
}
