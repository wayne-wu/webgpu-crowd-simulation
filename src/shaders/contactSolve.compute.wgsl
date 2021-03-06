////////////////////////////////////////////////////////////////////////////////
// Stability Solve & Short Range Collision 
////////////////////////////////////////////////////////////////////////////////

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> agentData_r : Agents;
@binding(2) @group(0) var<storage, read_write> agentData_w : Agents;
@binding(3) @group(0) var<storage, read_write> grid : Grid;
@binding(4) @group(0) var<storage, read_write> obstacleData : Obstacles;

@stage(compute) @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  var agent = agentData_r.agents[idx];
  var totalDx = vec3<f32>(0.0, 0.0, 0.0);
  var neighborCount = 0;

  if (agent.cell < 0){
    // ignore invalid cells
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
                                       nearRadius);

  let minX = bboxCorners[0];
  let minY = bboxCorners[1];
  let maxX = bboxCorners[2];
  let maxY = bboxCorners[3];

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

        var n = agent.xp - agent_j.xp;
        let d = length(n);
        if (d > nearRadius){
          continue;
        }

        let f = d - (agent.r + agent_j.r);
        if (f < 0.0) {
          // 4.2 Short Range Collision
          n = normalize(n);
          let w = agent.w / (agent.w + agent_j.w);
          var dx = -w * k_shortrange * f * n;

          if (friction) {
            // 4.2 Friction Contact (See 6.1 of https://mmacklin.com/uppfrta_preprint.pdf)
            // Add friction to slow down agents if collision is detected
            
            // Get corrected positions
            var xi = agent.xp + dx;
            var xj = agent_j.xp - dx;  // assumes mass are the same

            var d_rel = (xi - agent.x) - (xj - agent_j.x);
            var d_tan = d_rel - dot(d_rel, n) * n;  // project to tangential component
            var d_tan_norm = length(d_tan); 
            if(d_tan_norm >= mu_static * d) {
              d_tan = min(mu_kinematic * d/d_tan_norm, 1.0) * d_tan;
            }
            dx = dx + w * d_tan;

            // neighborCount = neighborCount + 1;
          }

          totalDx = totalDx + dx;
          neighborCount = neighborCount + 1;
        }
      }
    }
  }
  if (neighborCount > 0) {
    totalDx = avgCoefficient * totalDx / f32(neighborCount); 
    
    // Update position with correction
    agent.x = agent.x + totalDx;
    agent.xp = agent.xp + totalDx;
  }

  // 4.7 Obstacles Collision
  totalDx = vec3<f32>(0.0);
  neighborCount = 0;

  for (var j : u32 = 0u; j < arrayLength(&obstacleData.obstacles); j = j + 1u){
    obstacle_constraint(agent, obstacleData.obstacles[j], &neighborCount, &totalDx);
  }

  if (neighborCount > 0) {
    totalDx = avgCoefficient * totalDx / f32(neighborCount); 
    agent.x = agent.x + totalDx;
    agent.xp = agent.xp + totalDx;
  }

  // Store the new agent value
  agentData_w.agents[idx] = agent;
}
