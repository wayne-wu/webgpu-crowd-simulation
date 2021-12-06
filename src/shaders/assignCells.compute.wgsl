
////////////////////////////////////////////////////////////////////////////////
// Assign Cells Compute shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct SimulationParams {
  deltaTime : f32;
  avoidance : f32;
  numAgents : f32;
  gridWidth : f32;
};

struct Agent {
  x  : vec3<f32>;  // position + radius
  r  : f32;
  c  : vec4<f32>;  // color
  v  : vec3<f32>;  // velocity + inverse mass
  w  : f32;
  xp : vec3<f32>;  // planned/predicted position
  speed : f32;
  goal : vec3<f32>;
  cell : i32;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

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

  // get position relative to the start of the grid
  // (world origin is at the center of the grid)
  let pos = vec3<f32>(agent.x.x + (cellWidth * gridWidth / 2.0), 
                      agent.x.y, 
                      agent.x.z + (cellWidth * gridHeight / 2.0));
  
  var cell = vec2<i32>(i32(pos.x / cellWidth), i32(pos.z / cellWidth));

  var cellID : i32;

  // if outside grid, note that it's in an invalid cell
  if (cell.x >= i32(gridWidth) || cell.y >= i32(gridHeight) ||
      pos.x < 0.0 || pos.z < 0.0) {
    cellID = -1; 
  }
  else {
    cellID = cell.x + i32(gridWidth) * cell.y;
  }

  agent.cell = cellID;

  // Store the new agent value
  agentData.agents[idx] = agent;
}
