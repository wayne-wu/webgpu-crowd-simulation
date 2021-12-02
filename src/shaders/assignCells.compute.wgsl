
////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
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
  cell : u32;      // grid cell (linear form)
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
  var agent = agentData.agents[idx];

  let gridWidth = 50.0;
  let gridHeight = 50.0;
  let cellWidth = 2.0;

  let pos = vec3<f32>(agent.x.x + 50.0, agent.x.y, agent.x.z + 50.0);
  
  var cell = vec2<f32>(pos.x / cellWidth, pos.z / cellWidth);

  // if outside grid, belongs to first grid cell
  if (cell.x >= gridWidth || cell.y >= gridHeight ||
      cell.x < 0.0 || cell.y < 0.0) {
        cell = vec2<f32>(0.0, 0.0);
  }

  let cellID = u32(cell.x + gridWidth * cell.y);
  agent.cell = cellID;

  // change color for debugging purposes
  agent.c = vec4<f32>(cell.x / gridWidth, 0.0, cell.y / gridHeight, 1.0);

  // Store the new agent value
  agentData.agents[idx] = agent;
}
