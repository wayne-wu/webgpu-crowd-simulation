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

fn calcViscosity() -> vec3<f32>{

  return vec3<f32>(0.0, 0.0, 0.0);
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  // update velocity based on current position, corrected next position,
  // and XSPH viscosity
  agent.velocity = calcViscosity(); // + (nextPosition - currentPosition)

  // Store the new agent value
  // TODO uncomment this to actually set the velocity of our local agent
  //agentData.agents[idx] = agent;
}