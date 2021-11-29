////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
[[block]] struct SimulationParams {
  deltaTime : f32;
  seed : vec4<f32>;
  numAgents : u32;
};

struct Agent {
  x  : vec3<f32>;  // position + radius
  r  : f32;
  c  : vec4<f32>;  // color
  v  : vec3<f32>;  // velocity + inverse mass
  w  : f32;
  xp : vec3<f32>;  // planned/predicted position
  goal: vec3<f32>;
  cell : u32;      // grid cell (linear form)
};

[[block]] struct Agents {
  agents : array<Agent>;
};

//struct SortProxy {
//  index : u32;
//  cell : u32;
//}
//
//[[block]] struct SortProxies {
//  items : SortProxy;
//}

struct Cell {
  id : u32;
};

[[block]] struct GridCells {
  cells : array<Cell>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
//[[binding(2), group(0)]] var<storage, read_write> sortProxies : SortProxies;


//[[stage(compute), workgroup_size(64)]]
//fn scatter([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
//let idx = GlobalInvocationID.x;
//
//// TODO if idx is out of range, continue
//
//let item = sortProxies[i];
//
//if (item.index < 0){
//  // this was a padding item and has no useful data
//  return;
//}
//
//var agent = agentData[item.index];
//
//// I hope this syncs up all threads in the compute shader, not just
//// the workgroup, but I don't know that it actually does
//storageBarrier(); 
//
//}

fn swap(idx1 : u32, idx2 : u32) {
  //var tmp = agentData.agents[idx1];
  //agentData.agents[idx1] = agentData.agents[idx2];
  //agentData.agents[idx2] = tmp; 
  var tmp = agentData.agents[idx1].c[0];
  agentData.agents[idx1].c[0] = agentData.agents[idx2].c[0];
  agentData.agents[idx2].c[0] = tmp; 
}

fn agentlt(idx1 : u32, idx2 : u32) -> bool {
  //return agentData.agents[idx1].cell < agentData.agents[idx2].cell;
  return agentData.agents[idx1].c[0] < agentData.agents[idx2].c[0];
}

fn agentgt(idx1 : u32, idx2 : u32) -> bool {
  //return agentData.agents[idx1].cell > agentData.agents[idx2].cell;
  return agentData.agents[idx1].c[0] > agentData.agents[idx2].c[0];
}

//fn compAndSwap(idx1 : u32, idx2 : u32, dir : u32) {
//  if (dir == agentgt(idx1, idx2)){
//    swap(idx1, idx2);
//  }
//}


//fn bitonicMerge(start : u32, end : u32, direction : u32){
//  let mid : u32 = end - start;
//
//  for (let offset : u32 = mid + 1u; offset >= 1u; offset--){
//    for (let compDist : u32 = mid ) 
//    compAndSwap(start + )
//  }
//}
//
//fn bitonicSort(start : u32, end : u32, direction : u32){
//  let mid : u32 = end - start;
//
//  for (let i : u32 = start; i < start + mid; i++){
//    for (let compDist : u32 = mid; compDist > 0; compDist /= 2u){ 
//      compAndSwap(i, i + compDist, direction);
//    }
//  }
//}


[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x ;//+ (GlobalInvocationID.y * GlobalInvocationID.x);
  var agent = agentData.agents[idx];

  
  var i : u32;
  var j : u32;
  var k : u32;
  var l : u32;

  var n : u32 = 1024u;//sim_params.numAgents;



  // given an array arr of length n, this code sorts it in place
  // all indices run from 0 to n-1
  for (k = 2u; k <= n; k = k * 2u){ // k is doubled every iteration
    for (j = k/2u; j > 0u; j = j / 2u){ // j is halved at every iteration, with truncation of fractional parts
      //for (i = 0u; i < n; i = i + 1u){
        i = idx;
        l = i ^ j; 
        if (l > i){
          if (  (i & k) == 0u && agentgt(i,l) || (i & k) != 0u && agentlt(i, l)){
            swap(i, l);
            //storageBarrier();
          }
        }
      //}
    }
  }
}