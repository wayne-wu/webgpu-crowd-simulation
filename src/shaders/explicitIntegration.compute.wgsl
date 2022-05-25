////////////////////////////////////////////////////////////////////////////////
// Explicit Integration for Advecting the Agents
////////////////////////////////////////////////////////////////////////////////

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> agentData : Agents;

fn getVelocityFromPlanner(agent : Agent) -> vec3<f32> {
  // TODO: Implement a more complex planner
  return normalize(agent.goal - agent.x) * agent.speed;
}

@stage(compute) @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {

  let idx = GlobalInvocationID.x;
  var agent = agentData.agents[idx];

  if (idx >= u32(sim_params.numAgents)){
    return;
  }

  // velcity planning
  var vp = getVelocityFromPlanner(agent);

  // 4.1 Velocity Blending
  agent.v = (1.0 - ksi) * agent.v + ksi * vp;

  // explicit integration
  agent.xp = agent.x + sim_params.deltaTime * agent.v;

  // Store the new agent value
  agentData.agents[idx] = agent;
}
