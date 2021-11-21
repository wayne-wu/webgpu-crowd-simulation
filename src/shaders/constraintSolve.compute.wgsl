////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
let maxIterations : i32 = 6;
let t0 : f32 = 20.0;
let kUser : f32 = 1.0;  // TODO: User specified constant

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

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;


[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // TODO: 4.2 Friction Model (See 6.1 of https://mmacklin.com/uppfrta_preprint.pdf)

  // TODO: 4.4 Long Range Collision
  var itr = 0;
  loop {
    if (itr == maxIterations){ break; }
    
    var agent = agentData.agents[idx];
    var totalDx = vec3<f32>(0.0, 0.0, 0.0);
    var neighborCount = 0;

    for (var j : u32 = 0u; j < arrayLength(&agentData.agents); j = j + 1u) {
      if (idx == j) { continue; }
      
      let agent_j = agentData.agents[j];

      // if (distance(agent_j.xp, agent.xp) > neighborRadius) { continue; }

      let dt = sim_params.deltaTime;
      let r = agent.r + agent_j.r;
      let dist_xp_vec = agent.xp - agent_j.xp;  // distance vector xp
      let dist_x_vec = agent.x - agent_j.x;     // distance vector x

      let a = dot(dist_xp_vec, dist_xp_vec)/(dt*dt);
      let b = -dot(agent.x - agent_j.x, agent.xp, agent_j.xp)/dt;
      let c = dot(dist_x_vec, dist_x_vec) - r*r;

      // Compute exact time to collision
      let t = (b - sqrt(b*b - ac))/a;

      // Prune out invalid cases
      if (t <= 0 || t >= t0) { continue; }

      // Get time before and after collision
      let t_nocollision = dt * floor(t/dt);
      let t_collision = dt + t_nocollision;

      // Get collision and collision-free positions
      let xi_nocollision = agent.x + t_nocollision * agent.v;
      let xi_collision   = agent.x + t_collision * agent.v;
      let xj_nocollision = agent_j.x + t_nocollision * agent.v;
      let xj_collision   = agent_j.x + t_collision * agent.v;

      // Enforce collision free for x_collision using same as contactSolve
      var n = xi_collision - xj_collision;
      let d = length(n);

      if (d < agent.r + agent_j.r) {
        // Project Constraint
        n = normalize(n);
        var dx = -agent.w * d * n / (agent.w + agent_j.w);
        totalDx = totalDx + dx;
        neighborCount = neighborCount + 1;
      }

      // TODO: 4.5 Avoidance Model
      // let d = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);
      // let n = normalize(xi_nocollision - xj_nocollision);
      // let dt = d - dot(d,n)*n;
      // dx = dx in dt direction (tangential component only)
    }

    var stiffness = k * exp(-t_nocollision*t_nocollision/t0)
    // Constraint averaging: Not sure if this is needed yet
    totalDx = (1.0/f32(neighborCount)) * stiffness * totalDx; 

    // Update position with correction
    agent.xp = agent.xp + totalDx;

    // Store the new agent value
    agentData.agents[idx] = agent;

    // Sync Threads
    storageBarrier();
    workgroupBarrier();

    itr = itr + 1;
  }

  // Store the new agent value
  // TODO uncomment this to actually set the velocity of our local agent
  // agentData.agents[idx] = agent;
}