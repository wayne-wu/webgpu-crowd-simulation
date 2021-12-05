////////////////////////////////////////////////////////////////////////////////
// PBD Constraint Solving Compute Shader
////////////////////////////////////////////////////////////////////////////////

let maxIterations : i32 = 6;     // paper = 6
let t0 : f32 = 20.0;             // paper = 20
let tObstacle : f32 = 10.0;
let kUser : f32 = 0.15;          // paper = 0.24 [0-1]
let kObstacle : f32 = 0.5;
let avgCoefficient : f32 = 1.2;  // paper = 1.2  [1-2]

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
  nearNeighbors : array<u32, 20>; 
  farNeighbors : array<u32, 20>;
};

[[block]] struct Agents {
  agents : array<Agent>;
};

struct Obstacle {
  pos : vec3<f32>;
  rot : f32;
  scale : vec3<f32>;
};

[[block]] struct Obstacles {
  obstacles : array<Obstacle>;
};

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read_write> agentData : Agents;
[[binding(2), group(0)]] var<storage, read> obstacleData : Obstacles;

fn long_range_constraint(agent: Agent, agent_j: Agent, itr: i32, count: ptr<function, i32>, totalDx: ptr<function, vec3<f32>>)
{
  let dt = sim_params.deltaTime;

  let r = agent.r + agent_j.r;
  var r_sq = r * r;

  let dist = distance(agent.x, agent_j.x);
  if (dist < r) {
    r_sq = (r - dist) * (r - dist);
  }

  // relative displacement
  let x_ij = agent.x - agent_j.x;

  // relative velocity
  let v_ij = (1.0/dt) * (agent.xp - agent.x - agent_j.xp + agent_j.x);

  let a = dot(v_ij, v_ij);
  let b = -dot(x_ij, v_ij);
  let c = dot(x_ij, x_ij) - r_sq;
  var discr = b*b - a*c;
  if (discr <= 0.0 || abs(a) < 0.00001) { return; }

  discr = sqrt(discr);

  // Compute exact time to collision
  let t1 = (b - discr)/a;
  let t2 = (b + discr)/a;
  var t = select(t1, t2, t2 < t1 && t2 > 0.0);

  // Prune out invalid case
  if (t < 0.0 || t > t0) { return; }

  // Get time before and after collision
  let t_nocollision = dt * floor(t/dt);
  let t_collision = dt + t_nocollision;

  // Get collision and collision-free positions
  let xi_nocollision = agent.x + t_nocollision * agent.v;
  var xi_collision   = agent.x + t_collision * agent.v;
  let xj_nocollision = agent_j.x + t_nocollision * agent_j.v;
  var xj_collision   = agent_j.x + t_collision * agent_j.v;

  // Enforce collision free for x_collision using distance constraint
  var n = xi_collision - xj_collision;
  let d = length(n);

  let f = d - r;
  if (f < 0.0) {
    n = normalize(n);
    
    var k = kUser * exp(-t_nocollision*t_nocollision/t0);
    k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
    var dx = -agent.w * f * n / (agent.w + agent_j.w);

    // 4.5 Avoidance Model
    if (sim_params.avoidance == 1) {
      // get collision-free position
      xi_collision = xi_collision + dx;
      xj_collision = xj_collision - dx;

      // total relative displacement
      let d_vec = (xi_collision - xi_nocollision) - (xj_collision - xj_nocollision);

      // tangential relative displacement
      let d_tangent = d_vec - dot(d_vec, n)*n;
      dx = d_tangent;
    }

    // TODO: 4.2 Friction Model (See 6.1 of https://mmacklin.com/uppfrta_preprint.pdf)
    *totalDx = *totalDx + k * dx;
    *count = *count + 1;
  }
}

fn intersect_line(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, n: ptr<function, vec2<f32>>) -> f32
{
  let s1 = p1 - p0;
  let s2 = p3 - p2;

  var den = (-s2.x * s1.y + s1.x * s2.y);
  if (den < 0.00001) { return -1.0; }  // colinear

  den = 1.0/den;
  let s = (-s1.y * (p0.x - p2.x) + s1.x * (p0.y - p2.y)) * den;
  let t = ( s2.x * (p0.y - p2.y) - s2.y * (p0.x - p2.x)) * den;

  if (s > 0.0 && s < 1.0 && t > 0.0 && t < 1.0)
  {
    (*n).x = -s2.y;
    (*n).y = s2.x;
    return t;
  }

  return -1.0;
}

fn obstacle_constraint(agent: Agent, obstacle: Obstacle, itr: i32, count: ptr<function, i32>, totalDx: ptr<function, vec3<f32>>)
{
  // Create Model Matrix
  let c = cos(obstacle.rot);
  let s = sin(obstacle.rot);
  var m = mat4x4<f32>();
  m[0] = vec4<f32>(obstacle.scale.x*c, 0.0, -s, 0.0);
  m[1] = vec4<f32>(0.0, obstacle.scale.y, 0.0, 0.0);
  m[2] = vec4<f32>(s, 0.0, obstacle.scale.z*c, 0.0);
  m[3] = vec4<f32>(obstacle.pos, 1.0);

  // Get Corner Points in World Position (Cube)
  let l = 1.1;
  var p1 = (m * vec4<f32>(l,0.0,l,1.0)).xz;
  var p2 = (m * vec4<f32>(l,0.0,-l,1.0)).xz;
  var p3 = (m * vec4<f32>(-l,0.0,-l,1.0)).xz;
  var p4 = (m * vec4<f32>(-l,0.0,l,1.0)).xz;

  var v = (agent.xp - agent.x)/sim_params.deltaTime;
  var a0 = agent.xp.xz;
  var a1 = (agent.xp + tObstacle * v).xz;  // max look-ahead
  
  // Intersection test with the four edges
  var n_tmp : vec2<f32>;
  var n_min : vec2<f32>;
  var t_tmp : f32;
  var t_min : f32 = tObstacle;
  t_tmp = intersect_line(a0, a1, p1, p2, &n_tmp);
  if (t_tmp > 0.0 && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p2, p3, &n_tmp);
  if (t_tmp > 0.0 && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p3, p4, &n_tmp);
  if (t_tmp > 0.0 && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p4, p1, &n_tmp);
  if (t_tmp > 0.0 && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }

  if (t_min < 1.0) { 
    t_min = t_min * tObstacle;  // remap t_min to 0 to tObstacle
    
    //if(dot(v.xz, n_min) > 0.0) { n_min = -n_min; }  // flip the normal direction
    //var n = vec3<f32>(n_min.x, 0.0, n_min.y);  // contact normal

    // Use the radial normal as the contact normal so that there's some tangential velocity
    var n = normalize((agent.xp + t_min * v) - obstacle.pos);

    var k = kObstacle * exp(-t_min*t_min/tObstacle);
    k = 1.0 - pow(1.0 - k, 1.0/(f32(itr + 1)));
    var dx = k * n;
    *totalDx = *totalDx + dx;
    *count = *count + 1;
  } 
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;

  var itr = 0;
  loop {
    if (itr == maxIterations){ break; }
    
    var agent = agentData.agents[idx];
    var totalDx = vec3<f32>(0.0, 0.0, 0.0);
    var neighborCount = 0;
    let dt = sim_params.deltaTime;

    // 4.4 Long Range Collision
    for (var i : u32 = 0u; i < agent.farNeighbors[0]; i = i + 1u) {      
      long_range_constraint(agent, agentData.agents[agent.farNeighbors[1u+i]], itr, &neighborCount, &totalDx);
    }

    if (neighborCount > 0) {
      agent.xp = agent.xp + avgCoefficient * totalDx / f32(neighborCount);
    }

    totalDx = vec3<f32>(0.0);
    neighborCount = 0;

    // 4.7 Obstacles Avoidance
    for (var j : u32 = 0u; j < arrayLength(&obstacleData.obstacles); j = j + 1u){
      obstacle_constraint(agent, obstacleData.obstacles[j], itr, &neighborCount, &totalDx);
    }

    if (neighborCount > 0) {
      agent.xp = agent.xp + avgCoefficient * totalDx / f32(neighborCount);
    }

    // Store the new agent value
    agentData.agents[idx] = agent;

    // Sync Threads
    storageBarrier();
    workgroupBarrier();

    itr = itr + 1;
  }
}
