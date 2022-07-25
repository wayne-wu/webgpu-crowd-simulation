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
  var r = agent.r + agent_j.r;
  var r_sq = r * r;

  var dist = distance(agent.x, agent_j.x);
  if (dist < r) {
    r_sq = (r - dist) * (r - dist);
  }

  // relative displacement
  var x_ij = agent.x - agent_j.x;

  // relative velocity
  var v_ij = (1.0/dt) * (agent.xp - agent.x - agent_j.xp + agent_j.x);

  var a = dot(v_ij, v_ij);
  var b = -dot(x_ij, v_ij);
  var c = dot(x_ij, x_ij) - r_sq;
  var discr = b*b - a*c;
  if (discr < 0.0 || abs(a) < eps) { return; }

  discr = sqrt(discr);

  // Compute exact time to collision
  var t = (b - discr)/a;
  //const t2 = (b + discr)/a;
  //var t = select(t1, t2, t2 < t1 && t2 > 0.0);

  // Prune out invalid case
  if (t < eps || t > t0) { return; }

  // Get time before and after collision
  var t_nocollision = dt * floor(t/dt);
  var t_collision = dt + t_nocollision;

  // Get collision and collision-free positions
  var xi_nocollision = agent.x + t_nocollision * agent.v;
  var xi_collision   = agent.x + t_collision * agent.v;
  var xj_nocollision = agent_j.x + t_nocollision * agent_j.v;
  var xj_collision   = agent_j.x + t_collision * agent_j.v;

  // Enforce collision free for x_collision using distance constraint
  var n = xi_collision - xj_collision;
  var d = length(n);

  var f = d - r;
  if (f < 0.0) {
    n = normalize(n);
    
    var k = k_longrange * exp(-t_nocollision*t_nocollision/t0);
    k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
    var w = agent.w / (agent.w + agent_j.w);
    var dx = -w * f * n;

    // 4.5 Avoidance Model
    if (sim_params.avoidance == 1.0f) {
      // get collision-free position
      xi_collision = xi_collision + dx;
      xj_collision = xj_collision - dx;

      // total relative displacement
      var d_vec = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);

      // tangential relative displacement
      var d_tangent = d_vec - dot(d_vec, n)*n;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var idx = GlobalInvocationID.x;

  var itr = sim_params.iteration;
  var dt = sim_params.deltaTime;

  var agent = agentData_r.agents[idx];
  var totalDx = vec3<f32>(0.0, 0.0, 0.0);
  var neighborCount = 0;

  // 4.4 Long Range Collision
  if (agent.cell < 0){
    agentData_w.agents[idx] = agent;
    return;
  }

  var gridWidth = sim_params.gridWidth;
  var gridHeight = gridWidth;//sim_params.gridWidth;
  // TODO don't hardcode
  var cellWidth = 1000.0 / gridWidth;
  // compute cells that could conceivably contain neighbors
  var bboxCorners = getBBoxCornerCells(agent.x.x,
                                       agent.x.z,
                                       gridWidth,
                                       cellWidth,
                                       sim_params.farRadius);

  var minX = bboxCorners[0];
  var minY = bboxCorners[1];
  var maxX = bboxCorners[2];
  var maxY = bboxCorners[3];

  //for (var c : u32 = 0u; c < cellsToCheck; c = c + 1u ){
  for (var cellY = minY; cellY <= maxY; cellY = cellY + 1){
    if (cellY < 0 || cellY >= i32(gridHeight)){
      continue;
    }
    for (var cellX = minX; cellX <= maxX; cellX = cellX + 1){

      if (cellX < 0 || cellX >= i32(gridWidth)){
        continue;
      }
      var cellIdx = cell2dto1d(cellX, cellY, gridWidth);
      var cell : CellIndices = grid.cells[cellIdx];
      for (var i : u32 = cell.start; i <= cell.end; i = i + 1u) {

        if (idx == i) { 
          // ignore ourselves
          continue; 
        }
        var agent_j = agentData_r.agents[i];

        var dist = distance(agent.x, agent_j.x);

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
    var k = 1.0 - pow(1.0 - k_obstacle, 1.0/(f32(itr + 1)));
    agent.xp = agent.xp + avgCoefficient * k * totalDx / f32(neighborCount);
  }

  // Store the new agent value
  agentData_w.agents[idx] = agent;
}
