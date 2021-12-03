////////////////////////////////////////////////////////////////////////////////
// Finding Neighboring Agents Compute Shader
////////////////////////////////////////////////////////////////////////////////

let maxNeighbors : u32 = 20u;
let nearRadius : f32 = 2.0;
let farRadius : f32 = 5.0;

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
  goal : vec3<f32>;
  cell : i32;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

struct CellIndices {
  start : u32;
  end   : u32;
};

[[block]] struct Grid {
  cells : array<CellIndices>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read_write> grid : Grid;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

  var idx = GlobalInvocationID.x;

  if (idx >= u32(sim_params.numAgents)){
    return;
  }

  var agent = agentData.agents[idx];

  if (agent.cell < 0){
    // ignore invalid cells
    agent.c = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    agentData.agents[idx] = agent;
    return;
  }
  

  // compute neighbors
  var nearCount = 0u;
  var farCount = 0u;
  var near = array<u32, maxNeighbors>();
  var far = array<u32, maxNeighbors>();

  // NOTE these are also hardcoded in assignCells
  // if you change these here, change them there too
  //let gridWidth = 50;
  //let gridHeight = 50;
  //let cellWidth = 2;
  let gridWidth = i32(sim_params.gridWidth);
  let gridHeight = i32(sim_params.gridWidth);
  let cellWidth = i32(100 / gridWidth);

  //// TODO don't hardcode 9 cells 
  let cellsToCheck = 9u;
  var nearCellNums = array<i32, 9u>(
    agent.cell + gridWidth - 1, agent.cell + gridWidth, agent.cell + gridWidth + 1,
    agent.cell - 1, agent.cell, agent.cell+1, 
    agent.cell - gridWidth - 1, agent.cell - gridWidth, agent.cell - gridWidth + 1);
  //var nearCellNums = array<u32, 9u>(
  //  1386u, 1387u, 1388u, 
  //  1336u, 1337u, 1338u, 
  //  1286u, 1287u, 1288u, 
  //);
  //let cellsToCheck = 1u;
  //var nearCellNums = array<i32, 1u>(agent.cell);
  //let cellsToCheck = 100u;
  //var nearCellNums = array<i32, 100u>();
  //for (var i : u32 = 0u; i < cellsToCheck; i = i + 1u){

  //  nearCellNums[i] = i32(i);
  //}

  for (var c : u32 = 0u; c < cellsToCheck; c = c + 1u ){
    let cellIdx = nearCellNums[c];
    if (cellIdx < 0 || cellIdx >= gridWidth * gridHeight){
      continue;
    }
    let cell : CellIndices = grid.cells[cellIdx];
    // TODO make cell start/end
    //for (var j : u32 = cell.start; j <= cell.end; j = j + 1u) {
    for (var j : u32 = cell.start; j <= cell.end; j = j + 1u) {

      if (idx == j) { 
        // ignore ourselves
        //agent.c = vec4<f32>(1.0,1.0,1.0,1.0);
        continue; 
      }
        
      let agent_j = agentData.agents[j];
      let d = distance(agent_j.xp, agent.xp);

      // DEBUG
      //if (agent.cell == cellIdx){
      //  agent.c = vec4<f32>(1.0, 1.0, 1.0, 1.0);
      //}
      //else { 
      //  agent.c = vec4<f32>(1.0, 0.0, 1.0, 1.0);
      //
        
      if (d < nearRadius && nearCount < maxNeighbors - 1u) {
        nearCount = nearCount + 1u;
        near[nearCount] = j;
      }
      if (d < farRadius && farCount < maxNeighbors - 1u) {
        farCount = farCount + 1u;
        far[farCount] = j;
      }

      //if (nearCount == maxNeighbors) { 
      //  break; 
      //}
    }
  }

  //for (var j : u32 = 0u; j < arrayLength(&agentData.agents); j = j + 1u) {
  //  if (idx == j) { continue; }
  //    
  //  let agent_j = agentData.agents[j];
  //  let d = distance(agent_j.xp, agent.xp);
  //    
  //  if (d < farRadius && farCount < maxNeighbors - 1u) {
  //    farCount = farCount + 1u;
  //    far[farCount] = j;
  //  }

  //  if (farCount == maxNeighbors) { break; }
  //}

  near[0] = nearCount;
  far[0] = farCount;

  agent.nearNeighbors = near;
  agent.farNeighbors = far;

  let foo : f32 = f32(nearCount) / f32(maxNeighbors);
  agent.c = vec4<f32>(foo, 0.0, foo, 1.0);

  agentData.agents[idx] = agent;
}