// ----- Parameteres -----
let eps : f32 = 0.0001;
let t0 : f32 = 100.0;           // paper = 20
let tObstacle : f32 = 20.0;
let k_longrange : f32 = 0.24;   // paper = 0.24 [0-1]
let k_obstacle : f32 = 0.24;
let k_shortrange : f32 = 1.0;   // paper = 1.0 [0,1]
let k_avoid : f32 = 0.2;
let avgCoefficient : f32 = 1.2; // paper = 1.2 [1,2]
let farRadius : f32 = 5.0;
let nearRadius : f32 = 2.5;
let mu_static : f32 = 0.21;     // paper = 0.21
let mu_kinematic : f32 = 0.15;  // papser = ?
let ksi : f32 = 0.0385;         // paper = 0.0385
let xsph_c : f32 = 1.0;         // paper = 7.0
let xsph_h : f32 = 10.0;        // paper = 217.0 // the smoothing distance specified in the paper (assumes particles with radius 1)
let dir_blending : f32 = 0.8; 
let wall_radius : f32 = 0.5;  

let friction : bool = true;

// ----- Struct -----
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
  dir : vec3<f32>;
  group : f32;
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

struct Obstacle {
  pos : vec3<f32>;
  rot : f32;
  scale : vec3<f32>;
};

[[block]] struct Obstacles {
  obstacles : array<Obstacle>;
};


fn wall_constraint(agent: Agent, p0: vec2<f32>, p1: vec2<f32>, count: ptr<function, i32>, total_dx: ptr<function, vec3<f32>>) -> bool{
  let xp = agent.xp.xz;
  let a = xp - p0;
  let b = p1 - p0;
  let b_norm = normalize(b);
  let l = dot(a, b_norm);

  if(l < eps || l > length(b)) { return false; }

  let c = l * b_norm;

  let xj = p0 + c; 
  var n = a - c;
  let d = length(n);

  let r = agent.r + wall_radius;
  let f = d - r;
  if (f < 0.0)
  {
    n = normalize(n);
    var dx = - f * n;
    
    *total_dx = *total_dx + vec3<f32>(dx.x, 0.0, dx.y);
    *count = *count + 1;
    return true;
  }

  return false;
}


fn obstacle_constraint(agent: Agent, obstacle: Obstacle, count: ptr<function, i32>, total_dx: ptr<function, vec3<f32>>) {
  let c = cos(obstacle.rot);
  let s = sin(obstacle.rot);
  var m = mat4x4<f32>();
  m[0] = vec4<f32>(obstacle.scale.x*c, 0.0, -obstacle.scale.x*s, 0.0);
  m[1] = vec4<f32>(0.0, obstacle.scale.y, 0.0, 0.0);
  m[2] = vec4<f32>(obstacle.scale.z*s, 0.0, obstacle.scale.z*c, 0.0);
  m[3] = vec4<f32>(obstacle.pos, 1.0);

  // Get Corner Points in World Position (Cube)
  let l = 1.0;
  var p1 = (m * vec4<f32>(l,0.0,l,1.0)).xz;
  var p2 = (m * vec4<f32>(l,0.0,-l,1.0)).xz;
  var p3 = (m * vec4<f32>(-l,0.0,-l,1.0)).xz;
  var p4 = (m * vec4<f32>(-l,0.0,l,1.0)).xz;

  wall_constraint(agent, p1, p2, count, total_dx);
  wall_constraint(agent, p2, p3, count, total_dx);
  wall_constraint(agent, p3, p4, count, total_dx);
  wall_constraint(agent, p4, p1, count, total_dx);
}