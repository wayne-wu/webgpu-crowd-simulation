////////////////////////////////////////////////////////////////////////////////
// Stability Solve & Short Range Collision 
////////////////////////////////////////////////////////////////////////////////

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read> grid : Grid;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  var agent = agentData.agents[idx];
  var totalDx = vec3<f32>(0.0, 0.0, 0.0);
  var neighborCount = 0;

  if (agent.cell < 0){
    // ignore invalid cells
    agent.c = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    agentData.agents[idx] = agent;
    return;
  }

  let gridWidth = i32(sim_params.gridWidth);
  let gridHeight = i32(sim_params.gridWidth);
  // compute neighbors
  
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

      let agent_j = agentData.agents[i];

      var n = agent.xp - agent_j.xp;
      let d = length(n);
      if (d >= nearRadius){
        continue;
      }

      let f = d - (agent.r + agent_j.r);
      if (f < 0.0) {
        // 4.2 Short Range Collision
        n = normalize(n);
        let w = agent.w / (agent.w + agent_j.w);
        var dx = -w * k_shortrange * f * n;
        totalDx = totalDx + dx;
        neighborCount = neighborCount + 1;

        if (friction) {
          // 4.2 Friction Contact (See 6.1 of https://mmacklin.com/uppfrta_preprint.pdf)
          // Add friction to slow down agents if collision is detected
          
          // Get corrected positions
          var xi = agent.xp + dx;
          var xj = agent_j.xp - dx;  // assumes mass are the same

          var d_rel = (xi - agent.x) - (xj - agent_j.x);
          dx = d_rel - dot(d_rel, n) * n;  // project to tangential component
          var dx_norm = length(dx); 
          if(dx_norm >= mu_static * d) {
            dx = min(mu_kinematic * d/dx_norm, 1.0) * dx;
          }
          dx = w * dx;

          totalDx = totalDx + dx;
          neighborCount = neighborCount + 1;
        }
      }
    }

    if (neighborCount > 0) {
      // Constraint averaging: Not sure if this is needed yet
      totalDx = avgCoefficient * totalDx / f32(neighborCount); 
      
      // Update position with correction
      agent.x = agent.x + totalDx;
      agent.xp = agent.xp + totalDx;
    }
  }

  // Store the new agent value
  agentData.agents[idx] = agent;
}
