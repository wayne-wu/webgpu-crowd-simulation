// ----- Parameteres -----
let t0 : f32 = 20.0;            // paper = 20
let tObstacle : f32 = 10.0;
let k_longrange : f32 = 0.15;   // paper = 0.24 [0-1]
let k_obstacle : f32 = 1.0;
let k_shortrange : f32 = 1.0;   // paper = 1.0 [0,1]
let avgCoefficient : f32 = 1.2; // paper = 1.2 [1,2]
let farRadius : f32 = 5.0;
let nearRadius : f32 = 2.0;
let mu_static : f32 = 0.21;     // paper = 0.21
let mu_kinematic : f32 = 0.15;  // papser = ?
let ksi : f32 = 0.0385;         // paper = 0.0385
let xsph_c : f32 = 7.0;         // paper = 7.0
let xsph_h : f32 = 217.0;       // paper = 217.0 // the smoothing distance specified in the paper (assumes particles with radius 1)
let dir_blending : f32 = 0.8;   

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
  group : i32;
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