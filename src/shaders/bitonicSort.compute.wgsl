
  @id(1100) override j : u32;
  @id(1200) override k : u32;

  @binding(1) @group(0) var<storage, read_write> agentData : Agents;

  fn swap(idx1 : u32, idx2 : u32) {
    var tmp = agentData.agents[idx1];
    agentData.agents[idx1] = agentData.agents[idx2];
    agentData.agents[idx2] = tmp; 
  }

  fn agentlt(idx1 : u32, idx2 : u32) -> bool {
    return agentData.agents[idx1].cell < agentData.agents[idx2].cell;
  }

  fn agentgt(idx1 : u32, idx2 : u32) -> bool {
    return agentData.agents[idx1].cell > agentData.agents[idx2].cell;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var idx = GlobalInvocationID.x ;
    
    var l = idx ^ j; 
    if (l > idx){
      if (  ((idx & k) == 0u && agentgt(idx,l)) || ((idx & k) != 0u && agentlt(idx, l))){
        swap(idx, l);
      }
    }
  }

