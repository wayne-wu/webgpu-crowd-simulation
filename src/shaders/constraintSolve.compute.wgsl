////////////////////////////////////////////////////////////////////////////////
// PBD Constraint Solving Compute Shader
////////////////////////////////////////////////////////////////////////////////

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read> agentData_r : Agents;
[[binding(2), group(0)]] var<storage, write> agentData_w : Agents;
[[binding(3), group(0)]] var<storage, read> grid : Grid;
[[binding(4), group(0)]] var<storage, read> obstacleData : Obstacles;

fn long_range_constraint(agent: Agent, agent_j: Agent, itr: i32, count: ptr<function, i32>, totalDx: ptr<function, vec3<f32>>)
{
  let dt = sim_params.deltaTime;

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
  if (discr < 0.0 || abs(a) < eps) { return; }

  discr = sqrt(discr);

  // Compute exact time to collision
  let t = (b - discr)/a;
  //let t2 = (b + discr)/a;
  //var t = select(t1, t2, t2 < t1 && t2 > 0.0);

  // Prune out invalid case
  if (t < eps || t > t0) { return; }

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
    
    var k = k_longrange * exp(-t_nocollision*t_nocollision/t0);
    k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
    let w = agent.w / (agent.w + agent_j.w);
    var dx = -w * f * n;

    // 4.5 Avoidance Model
    if (sim_params.avoidance == 1.0f) {
      // get collision-free position
      xi_collision = xi_collision + dx;
      xj_collision = xj_collision - dx;

      // total relative displacement
      let d_vec = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);

      // tangential relative displacement
      let d_tangent = d_vec - dot(d_vec, n)*n;
      dx = w * d_tangent;
    }

    *totalDx = *totalDx + k * dx;
    *count = *count + 1;
  }
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  var itr = sim_params.iteration;

  var agent = agentData_r.agents[idx];
  var totalDx = vec3<f32>(0.0, 0.0, 0.0);
  var neighborCount = 0;
  let dt = sim_params.deltaTime;

  // 4.4 Long Range Collision
  if (agent.cell < 0){
    // ignore invalid cells
    agent.c = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    agentData_w.agents[idx] = agent;
    return;
  }

  let gridWidth = i32(sim_params.gridWidth);
  let gridHeight = i32(sim_params.gridWidth);
  // compute neighbors
  var nearCount = 0u;
  var farCount = 0u;
  //// TODO don't hardcode 9 cells 
  let cellsToCheck = 9u;
  var nearCellNums = array<i32, 9u>(
    agent.cell + gridWidth - 1, agent.cell + gridWidth, agent.cell + gridWidth + 1,
    agent.cell - 1, agent.cell, agent.cell+1, 
    agent.cell - gridWidth - 1, agent.cell - gridWidth, agent.cell - gridWidth + 1);

  for (var c : u32 = 0u; c < cellsToCheck; c = c + 1u ){
    let cellIdx = nearCellNums[c];
    if (cellIdx < 0 || cellIdx >= gridWidth * gridHeight){
      continue;
    }
    let cell : CellIndices = grid.cells[cellIdx];
    for (var i : u32 = cell.start; i <= cell.end; i = i + 1u) {

      if (idx == i) { 
        // ignore ourselves
        continue; 
      }
      let agent_j = agentData_r.agents[i];

      let dist = distance(agent.x, agent_j.x);

      if (dist >= farRadius){
        continue;
      }

      long_range_constraint(agent, agent_j, itr, &neighborCount, &totalDx);
    }
  }

  if (neighborCount > 0) {
    agent.xp = agent.xp + avgCoefficient * totalDx / f32(neighborCount);
  }

  // 4.7 Obstacles Collision
  totalDx = vec3<f32>(0.0);
  neighborCount = 0;
  
  for (var j : u32 = 0u; j < arrayLength(&obstacleData.obstacles); j = j + 1u){
    obstacle_constraint(agent, obstacleData.obstacles[j], &neighborCount, &totalDx);
  }

  if (neighborCount > 0) {
    let k = 1.0 - pow(1.0 - k_obstacle, 1.0/(f32(itr + 1)));
    agent.xp = agent.xp + avgCoefficient * k * totalDx / f32(neighborCount);
  }

  // Store the new agent value
  agentData_w.agents[idx] = agent;
}
