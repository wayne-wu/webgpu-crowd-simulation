////////////////////////////////////////////////////////////////////////////////
// Explicit Integration for Advecting the Agents
////////////////////////////////////////////////////////////////////////////////

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> agentData : Agents;

fn getVelocityFromPlanner(agent : Agent) -> vec3<f32> {
  // TODO: Implement a more complex planner
  return normalize(agent.goal - agent.x) * agent.speed;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {

  var idx = GlobalInvocationID.x;

  if (idx >= u32(sim_params.numAgents)){
    return;
  }

  var agent = agentData.agents[idx];

  // velcity planning
  var vp = getVelocityFromPlanner(agent);

  // 4.1 Velocity Blending
  agent.v = (1.0 - ksi) * agent.v + ksi * vp;

  // explicit integration
  agent.xp = agent.x + sim_params.deltaTime * agent.v;

  if (distance(agent.x, agent.goal) < 1.0){
    //   // if we're close enough to the goal, (somewhat) smoothly place them 
    //   // on the goal and set the cell to invalid so future computation largely
    //   // ignores them 
    //   agent.x = ((agent.x * 10.0) + agent.goal) / 11.0;
    agent.cell = -2;
    agent.c = rainbowCycle(sim_params.tick); 
  }
  else {
    var gridWidth = sim_params.gridWidth;
    var gridHeight = sim_params.gridWidth;
    var cellWidth = 1000.0 / gridWidth;

    var posCellSpace = worldSpacePosToCellSpace(agent.x.x, 
                                                agent.x.z, 
                                                gridWidth, 
                                                cellWidth);

    var cellXY = cellSpaceToCell2d(posCellSpace.x, posCellSpace.y, cellWidth);

    agent.cell = cell2dto1d(cellXY.x, cellXY.y, gridWidth);
    if (cellXY.x >= i32(gridWidth) || cellXY.y >= i32(gridHeight) ||
        posCellSpace.x < 0.0 || posCellSpace.y < 0.0) {
      agent.cell = -1; 
      agent.c = vec4<f32>(0.5, 0.5, 0.5, 1.0);
    }
  }

  // Store the new agent value
  agentData.agents[idx] = agent;
}
