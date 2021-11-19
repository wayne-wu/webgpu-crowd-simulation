////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct SimulationParams {
  deltaTime : f32;
  seed : vec4<f32>;
};

struct Agent {
  position : vec3<f32>;
  lifetime : f32;
  color    : vec4<f32>;
  velocity : vec3<f32>;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

struct Goal {
  vel : vec3<f32>;
};

[[block]] struct GoalData {
  goals : array<Goal>;
};

struct Cell {
  id : u32;
};

[[block]] struct GridCells {
  cells : array<Cell>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read_write> goalData : GoalData;
[[binding(3), group(0)]] var<storage, read_write> gridCell : GridCells;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // loop at line 9 in algorithm 1 of the paper
  let maxIterations : u32 = 1u; // TODO set to number of stability iterations
  //for (var i : u32 = 0u; i < maxIterations; i++){
  //  // compute position correction deltaX

  //  // current position += deltaX

  //  // planned next position += deltaX
  //}

  //// loop at line 16 in algorithm 1 of the paper
  //maxIterations = 1; // TODO set to whatever the paper means by "max iterations"
  //for (var i : u32 = 0u; i < maxIterations; i++){
  //  // compute position correction deltaX

  //  // planned next position += deltaX
  //}

  // Store the new agent value
  agentData.agents[idx] = agent;
}