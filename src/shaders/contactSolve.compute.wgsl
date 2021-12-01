////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
let maxIterations : i32 = 1;
let stiffness : f32 = 1.0;

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

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  // 4.2 Short Range Collision
  var itr = 0;
  loop {
    if (itr == maxIterations){ break; }
    
    var agent = agentData.agents[idx];
    var totalDx = vec3<f32>(0.0, 0.0, 0.0);
    var neighborCount = 0;

    for (var i : u32 = 0u; i < agent.nearNeighbors[0]; i = i + 1u) {      
      let agent_j = agentData.agents[agent.nearNeighbors[1u+i]];

      var n = agent.xp - agent_j.xp;
      let d = length(n);

      let f = d - (agent.r + agent_j.r);
      if (f < 0.0) {
        // Project Constraint
        n = normalize(n);
        var dx = -agent.w * stiffness * f * n / (agent.w + agent_j.w);
        totalDx = totalDx + dx;
        neighborCount = neighborCount + 1;
      }
    }

    if (neighborCount > 0) {
      // Constraint averaging: Not sure if this is needed yet
      totalDx = totalDx / f32(neighborCount); 
      
      // Update position with correction
      agent.x = agent.x + totalDx;
      agent.xp = agent.xp + totalDx;
    }

    // Store the new agent value
    agentData.agents[idx] = agent;

    // Sync Threads
    storageBarrier();
    workgroupBarrier();

    itr = itr + 1;
  }
}