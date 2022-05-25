////////////////////////////////////////////////////////////////////////////////
// PBD Constraint Solving Compute Shader
////////////////////////////////////////////////////////////////////////////////

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> agentData_r : Agents;
@binding(2) @group(0) var<storage, read_write> agentData_w : Agents;
@binding(3) @group(0) var<storage, read_write> grid : Grid;
@binding(4) @group(0) var<storage, read_write> obstacleData : Obstacles;

fn long_range_constraint(agent: Agent, 
                         agent_j: Agent, 
                         itr: i32, 
                         dt : f32,
                         count: ptr<function, i32>, 
                         totalDx: ptr<function, vec3<f32>>)
{
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

      // NOTE: https://github.com/tomerwei/pbd-crowd-sim/blob/master/src/crowds_cpu_orginal.cpp#L1584
      // The author seems to be adding tangential component back to
      // the original dx which does yield better result. (Not 100% sure why)
      
      // Email from Dr. Weiss:
      // 1) The avoidance model is the same as the LR algorithm, 
      // except that it only maintains the tangential component of collision avoidance behavior.
      // The avoidance behavior keeps only the tangential component of the friction contact. 
      // 2) In long range we have both tangential + normal contact vectors influencing the positional corrections. 
      // In the avoidance avoidance behavior, only the tangential component contributes to the positional correction. 

      dx = dx + w * d_tangent;
      *count = *count + 1;
    }

    *totalDx = *totalDx + k * dx;
    *count = *count + 1;
  }
}

@stage(compute) @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  var itr = sim_params.iteration;
  let dt = sim_params.deltaTime;

  var agent = agentData_r.agents[idx];
  var totalDx = vec3<f32>(0.0, 0.0, 0.0);
  var neighborCount = 0;

  // 4.4 Long Range Collision
  if (agent.cell < 0){
    agentData_w.agents[idx] = agent;
    return;
  }

  let gridWidth = sim_params.gridWidth;
  let gridHeight = gridWidth;//sim_params.gridWidth;
  // TODO don't hardcode
  let cellWidth = 1000.0 / gridWidth;
  // compute cells that could conceivably contain neighbors
  let bboxCorners = getBBoxCornerCells(agent.x.x,
                                       agent.x.z,
                                       gridWidth,
                                       cellWidth,
                                       sim_params.farRadius);

  let minX = bboxCorners[0];
  let minY = bboxCorners[1];
  let maxX = bboxCorners[2];
  let maxY = bboxCorners[3];

  //for (var c : u32 = 0u; c < cellsToCheck; c = c + 1u ){
  for (var cellY = minY; cellY <= maxY; cellY = cellY + 1){
    if (cellY < 0 || cellY >= i32(gridHeight)){
      continue;
    }
    for (var cellX = minX; cellX <= maxX; cellX = cellX + 1){

      if (cellX < 0 || cellX >= i32(gridWidth)){
        continue;
      }
      let cellIdx = cell2dto1d(cellX, cellY, gridWidth);
      let cell : CellIndices = grid.cells[cellIdx];
      for (var i : u32 = cell.start; i <= cell.end; i = i + 1u) {

        if (idx == i) { 
          // ignore ourselves
          continue; 
        }
        let agent_j = agentData_r.agents[i];

        let dist = distance(agent.x, agent_j.x);

        if (dist > sim_params.farRadius){
          continue;
        }

        long_range_constraint(agent, agent_j, itr, dt, &neighborCount, &totalDx);
      }
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
