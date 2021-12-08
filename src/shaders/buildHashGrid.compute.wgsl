////////////////////////////////////////////////////////////////////////////////
// Build Hash Grid - figure out start/end indices of cells in sorted agent array
////////////////////////////////////////////////////////////////////////////////

[[block]] struct SimulationParams {
  deltaTime : f32;
  avoidance : f32;
  numAgents : f32;
  gridWidth : f32;
  iteration : i32;
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
  // build Grid, an array of start/end indicies that indicate where
  // each grid cell starts/ends on an array of agents sorted by cell

  let idx = GlobalInvocationID.x;

  if (idx >= u32(sim_params.numAgents) - 2u){
    // the idx doesn't correspond to a valid agent
    return;
  }

  // get the agent that corresponds to this index
  var agentL = agentData.agents[idx];
  var agentR = agentData.agents[idx + 1u];
  
  if (idx == 0u){
    // if this is index 0, it has to be the start
    // of the relevant cell
    grid.cells[agentL.cell].start = idx;
  }
  if (idx == u32(sim_params.numAgents) - 1u){
    // if this is the last index, it has to be the end
    // of the relevant cell
    grid.cells[agentR.cell].end = idx + 1u;
  }

  if (agentL.cell != agentR.cell){
    // if the two contiguous agents have different
    // cells, that means the left is the end of its block
    // of cells and the right is the start of a new one.
    // Note: searching this way, each cell's start or end
    // is only edited by one thread, so we avoid race conditions.
    if (agentL.cell != -1){
      grid.cells[agentL.cell].end = idx;
    }
    if (agentR.cell != -1){
      grid.cells[agentR.cell].start = idx + 1u;
    }
  }
}
