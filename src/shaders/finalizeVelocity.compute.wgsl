////////////////////////////////////////////////////////////////////////////////
// Finalize Velocity Compute Shader
////////////////////////////////////////////////////////////////////////////////

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read> agentData_r : Agents;
[[binding(2), group(0)]] var<storage, write> agentData_w : Agents;
[[binding(3), group(0)]] var<storage, read> grid : Grid;

fn getW(d : f32) -> f32 {
    var w = 0.0; // poly6 smoothing kernel

    if (0.0 <= d && d <= xsph_h) {
        w = 315.0 / (64.0 * 3.14159 * pow(xsph_h, 9.0));
        w = w * pow( pow(xsph_h, 2.0) - pow(d, 2.0), 3.0 );
    }
    return w;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData_r.agents[idx];

  // PBD: Get new velocity from corrected position
  var last_v = agent.v;
  agent.v = (agent.xp - agent.x)/sim_params.deltaTime;

  // 4.3 Cohesion
  // update velocity to factor in viscosity
  var velAvg = vec3<f32>(0.0); // weighted average of all the velocity differences

  if (agent.cell < 0){
    // ignore invalid cells
    agent.c = vec4<f32>(0.5, 0.5, 0.5, 1.0);
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

      var neighbor = agentData_r.agents[i];
      var d = distance(agent.x, neighbor.x);  // Should this be xp or x?
      if (d >= nearRadius){
        continue;
      }
      var w = getW(d*d);
      velAvg = velAvg + (agent.v - neighbor.v) * w;
    }
  }
  agent.v = agent.v + xsph_c * velAvg;

  // 4.6 Maximum Speed and Acceleration Limiting

  let v_dir = normalize(agent.v);
  let maxSpeed : f32 = agent.speed;
  if(length(agent.v) > maxSpeed){
    agent.v = maxSpeed * v_dir;
  }

  // Set new position to be the corrected position
  // Reintegrate here so that the position doesn't jump between frames
  agent.x = agent.x + agent.v * sim_params.deltaTime;

  agent.dir = dir_blending * normalize(agent.dir) + (1.0 - dir_blending) * v_dir;

  // Store the new agent value
  agentData_w.agents[idx] = agent;
}
