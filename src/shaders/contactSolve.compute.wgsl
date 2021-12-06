////////////////////////////////////////////////////////////////////////////////
// Stability Solve & Short Range Collision 
////////////////////////////////////////////////////////////////////////////////

let maxIterations : i32 = 1;  // paper = 1
let stiffness : f32 = 1.0;  // paper = 1.0 [0,1]
let avgCoefficient : f32 = 1.2;  // paper = 1.2 [1,2]
let nearRadius : f32 = 2.0;

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
  let idx = GlobalInvocationID.x;

  // 4.2 Short Range Collision
  var itr = 0;
  loop {
    if (itr == maxIterations){ break; }
    
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

        let agent_j = agentData.agents[i];

        var n = agent.xp - agent_j.xp;
        let d = length(n);
        if (d >= nearRadius){
          continue;
        }

        let f = d - (agent.r + agent_j.r);
        if (f < 0.0) {
          // Project Constraint
          n = normalize(n);
          var dx = -agent.w * stiffness * f * n / (agent.w + agent_j.w);
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

    // Store the new agent value
    agentData.agents[idx] = agent;

    // Sync Threads
    //storageBarrier();
   // workgroupBarrier();

    itr = itr + 1;
  }
}
