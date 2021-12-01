////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////

let maxSpeed : f32 = 2.0;
let maxNeighbors : u32 = 20u;

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
  nearNeighbors : array<u32, 20>; 
  farNeighbors : array<u32, 20>;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

fn getW(d : f32) -> f32 {
    var h = 7.0; // the smoothing distance specified in the paper (assumes particles with radius 1)
    var w = 0.0; // poly6 smoothing kernel

    if (0.0 <= d && d <= h) {
        w = 315.0 / (64.0 * 3.14159 * pow(h, 9.0));
        w = w * pow( pow(h, 2.0) - pow(d, 2.0), 3.0 );
    }
    return w;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // PBD: Get new velocity from corrected position
  agent.v = (agent.xp - agent.x)/sim_params.deltaTime;

  // 4.3 Cohesion
  // update velocity to factor in viscosity
  var c = 1.0; // based on paper
  var velAvg = vec3<f32>(0.0); // weighted average of all the velocity differences

  for (var i : u32 = 0u; i < agent.nearNeighbors[0]; i = i + 1u){
    var neighbor = agentData.agents[agent.nearNeighbors[1u+i]];
    var d = distance(agent.x, neighbor.x);  // Should this be xp or x?
    var w = getW(d*d);
    velAvg = velAvg + (agent.v - neighbor.v) * w;
  }
  agent.v = agent.v + c * velAvg;

  // 4.6 Maximum Speed and Acceleration Limiting
  if(length(agent.v) > maxSpeed){
    agent.v = maxSpeed * normalize(agent.v);
  }

  // Set new position to be the corrected position
  agent.x = agent.xp;
  
  // Store the new agent value
  agentData.agents[idx] = agent;
}