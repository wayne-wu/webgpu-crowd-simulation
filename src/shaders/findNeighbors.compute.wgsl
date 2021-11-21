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
};

[[block]] struct Agents {
  agents : array<Agent>;
};

struct Goal {
  vel : vec3<f32>;
};

[[block]] struct GoalData {
  goals : array<Goal>;
};

[[block]] struct Neighbors {
  neighbors: array<u32>;  // up to 20 
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read_write> goalData : GoalData;
[[binding(4), group(0)]] var<storage, read_write> neighborData : Neighbors;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

  var idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];
  var radius = 5.0;

  var neighborIdx = 0;
  // loop through agents, find neighbors, and save their indices to the neighbor buffer
  for (var i : u32 = 0u; i < arrayLength(&agentData.agents); i = i + 1u) {
    var potentialNeighbor = agentData.agents[i];
    if (distance(potentialNeighbor.x, agent.x) < radius) {
      agent.neighbors[neighborIdx] = i;
      neighborIdx = neighborIdx + 1;
      // if (idx == 0u) {
      //   potentialNeighbor.c = vec4<f32>(0.0, 1.0, 0.0, 1.0);
      // }
    }   
  }

  // if (idx == 0u){
  //   agent.c = vec4<f32>(0.0, 1.0, 0.0, 1.0);
  // }
  // if (distance(agent.x, agentData.agents[0].x) < radius) {
  //   agent.c = vec4<f32>(0.0, 1.0, 0.0, 1.0);
  // }
  // else{
  //   agent.c = vec4<f32>(0.9, 0.9, 0.9, 1.0);
  // }

  // Store the new agent value
  agentData.agents[idx] = agent;
}