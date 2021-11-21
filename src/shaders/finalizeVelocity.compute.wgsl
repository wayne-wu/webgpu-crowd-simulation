////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////

let maxSpeed : f32 = 2.0;

[[block]] struct SimulationParams {
  deltaTime : f32;
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
};

[[block]] struct Agents {
  agents : array<Agent>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

fn calcViscosity() -> vec3<f32>{

  return vec3<f32>(0.0, 0.0, 0.0);
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // PBD: Get new velocity from corrected position
  agent.v = (agent.xp - agent.x)/sim_params.deltaTime;
  
  // TODO: 4.3 Cohesion
  // update velocity based on current position, corrected next position,
  // and XSPH viscosity
  // agent.v = calcViscosity(); // + (nextPosition - currentPosition)

  // 4.6 Maximum Speed and Acceleration Limiting
  if(length(agent.v) > maxSpeed){
    agent.v = maxSpeed * normalize(agent.v);
  }

  // Set new position to be the corrected position
  agent.x = agent.xp;
  
  // Store the new agent value
  agentData.agents[idx] = agent;
}