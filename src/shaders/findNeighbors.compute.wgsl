////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
let maxNeighbors : u32 = 20u;
let nearRadius : f32 = 2.0;
let farRadius : f32 = 5.0;

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
  nearNeighbors : array<u32, maxNeighbors>; 
  farNeighbors : array<u32, maxNeighbors>;
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

  // compute neighbors
  var nearCount = 0u;
  var farCount = 0u;
  var near = array<u32, maxNeighbors>();
  var far = array<u32, maxNeighbors>();

  for (var j : u32 = 0u; j < arrayLength(&agentData.agents); j = j + 1u) {
    if (idx == j) { continue; }
      
    let agent_j = agentData.agents[j];
    let d = distance(agent_j.xp, agent.xp);
      
    if (d < nearRadius && nearCount < maxNeighbors - 1u) {
      nearCount = nearCount + 1u;
      near[nearCount] = j;
    }
    else { 
      if (d < farRadius && farCount < maxNeighbors - 1u) {
        farCount = farCount + 1u;
        far[farCount] = j;
      }
    }

    if (nearCount == maxNeighbors && farCount == maxNeighbors) { break; }
  }

  near[0] = nearCount;
  far[0] = farCount;

  agent.nearNeighbors = near;
  agent.farNeighbors = far;

  agentData.agents[idx] = agent;
}