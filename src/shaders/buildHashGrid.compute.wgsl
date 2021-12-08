////////////////////////////////////////////////////////////////////////////////
// Build Hash Grid - figure out start/end indices of cells in sorted agent array
////////////////////////////////////////////////////////////////////////////////

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
