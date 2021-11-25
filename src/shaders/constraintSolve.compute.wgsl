////////////////////////////////////////////////////////////////////////////////
// PBD Constraint Solving Compute Shader
////////////////////////////////////////////////////////////////////////////////

let maxIterations : i32 = 6;     // paper = 6
let t0 : f32 = 20.0;             // paper = 20
let kUser : f32 = 0.15;          // paper = 0.24 [0-1]
let avgCoefficient : f32 = 1.2;  // paper = 1.2  [1-2]

let avoidance : bool = true;

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


[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  // 4.4 Long Range Collision
  var itr = 0;
  loop {
    if (itr == maxIterations){ break; }
    
    var agent = agentData.agents[idx];
    var totalDx = vec3<f32>(0.0, 0.0, 0.0);
    var neighborCount = 0;
    let dt = sim_params.deltaTime;

    for (var i : u32 = 0u; i < agent.farNeighbors[0]; i = i + 1u) {      
      let agent_j = agentData.agents[agent.farNeighbors[1u+i]];

      let r = agent.r + agent_j.r;
      var r_sq = r * r;

      let dist = distance(agent.x, agent_j.x);
      if (dist < r) {
        r_sq = (r - dist) * (r - dist);
      }

      // relative displacement
      let x_ij = agent.x - agent_j.x;

      // relative velocity
      let v_ij = (1.0/dt) * (agent.xp - agent.x - agent_j.xp + agent_j.x);

      let a = dot(v_ij, v_ij);
      let b = -dot(x_ij, v_ij);
      let c = dot(x_ij, x_ij) - r_sq;
      var discr = b*b - a*c;
      if (discr <= 0.0 || abs(a) < 0.00001) { continue; }

      discr = sqrt(discr);

      // Compute exact time to collision
      let t1 = (b - discr)/a;
      let t2 = (b + discr)/a;
      var t = select(t1, t2, t2 < t1 && t2 > 0.0);

      // Prune out invalid case
      if (t < 0.0 || t > t0) { continue; }

      // Get time before and after collision
      let t_nocollision = dt * floor(t/dt);
      let t_collision = dt + t_nocollision;

      // Get collision and collision-free positions
      let xi_nocollision = agent.x + t_nocollision * agent.v;
      var xi_collision   = agent.x + t_collision * agent.v;
      let xj_nocollision = agent_j.x + t_nocollision * agent_j.v;
      var xj_collision   = agent_j.x + t_collision * agent_j.v;

      // Enforce collision free for x_collision using distance constraint
      var n = xi_collision - xj_collision;
      let d = length(n);

      let f = d - r;
      if (f < 0.0) {
        n = normalize(n);
        
        var k = kUser * exp(-t_nocollision*t_nocollision/t0);
        k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
        var dx = -agent.w * f * n / (agent.w + agent_j.w);

        // 4.5 Avoidance Model
        if (sim_params.avoidance == 1) {
          // get collision-free position
          xi_collision = xi_collision + dx;
          xj_collision = xj_collision - dx;

          // total relative displacement
          let d_vec = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);

          // tangential relative displacement
          let d_tangent = d_vec - dot(d_vec, n)*n;
          dx = d_tangent;
        }

        // TODO: 4.2 Friction Model (See 6.1 of https://mmacklin.com/uppfrta_preprint.pdf)
        totalDx = totalDx + k * dx;
        neighborCount = neighborCount + 1;
      }
    }

    if (neighborCount > 0) {
      // Update position with correction
      agent.xp = agent.xp + avgCoefficient * totalDx / f32(neighborCount);
    }

    // Store the new agent value
    agentData.agents[idx] = agent;

    // Sync Threads
    storageBarrier();
    workgroupBarrier();

    // 4.2 Friction Model


    itr = itr + 1;
  }
}
