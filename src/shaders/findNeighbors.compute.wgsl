////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
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
  neighbors : array<u32, 20>; // neighbors, max 20
};

[[block]] struct Agents {
  agents : array<Agent>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

  var idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];
  var radius = 1.0;

  // compute neighbors
  var neighborCount = 0;
  var neighbors = array<u32, 20>();

  for (var j : u32 = 0u; j < arrayLength(&agentData.agents); j = j + 1u) {
    if (idx == j) { continue; }
      
    let agent_j = agentData.agents[j];
      
    if (distance(agent_j.x, agent.x) > radius) { continue; }
    if (neighborCount >= 20) { continue; }

    neighborCount = neighborCount + 1;
    neighbors[neighborCount] = j;
  }

  // Store the new agent value
  agent.neighbors = neighbors;
  agentData.agents[idx] = agent;
}