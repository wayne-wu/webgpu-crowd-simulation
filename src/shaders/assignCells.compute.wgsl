
////////////////////////////////////////////////////////////////////////////////
// Assign Cells Compute shader
////////////////////////////////////////////////////////////////////////////////

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  // Calculate which grid cell each agent is in and store it

  let idx = GlobalInvocationID.x;
  if (idx >= u32(sim_params.numAgents)){
    return;
  }

  var agent = agentData.agents[idx];

  let gridWidth = sim_params.gridWidth;
  let gridHeight = sim_params.gridWidth;
  let cellWidth = 1000.0 / gridWidth;

  let posCellSpace = worldSpacePosToCellSpace(agent.x.x, 
                                              agent.x.z, 
                                              gridWidth, 
                                              cellWidth);

  let cellXY = cellSpaceToCell2d(posCellSpace.x, posCellSpace.y, cellWidth);

  var cellID = cell2dto1d(cellXY.x, cellXY.y, gridWidth);

  if (cellXY.x >= i32(gridWidth) || cellXY.y >= i32(gridHeight) ||
      posCellSpace.x < 0.0 || posCellSpace.y < 0.0) {
    cellID = -1; 
  }

  agent.cell = cellID;

  // Store the new agent value
  agentData.agents[idx] = agent;
}
