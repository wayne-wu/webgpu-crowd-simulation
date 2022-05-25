// ----- Parameteres -----
let eps : f32 = 0.0001;
let t0 : f32 = 20.0;            // paper = 20
let tObstacle : f32 = 20.0;
let k_longrange : f32 = 0.15;   // paper = 0.24 [0-1]
let k_obstacle : f32 = 0.24;
let k_shortrange : f32 = 1.0;   // paper = 1.0 [0,1]
let k_avoid : f32 = 0.2;
let avgCoefficient : f32 = 1.2; // paper = 1.2 [1,2]
// let farRadius : f32 = 6.0;   // Promoted to GUI
let nearRadius : f32 = 2.0;
let cohesionRadius : f32 = 5.0;
let mu_static : f32 = 0.21;     // paper = 0.21
let mu_kinematic : f32 = 0.10;  // papser = ?
let ksi : f32 = 0.0385;         // paper = 0.0385
let xsph_c : f32 = 7.0;         // paper = 7.0
let xsph_h : f32 = 217.0;       // paper = 217.0 // the smoothing distance specified in the paper (assumes particles with radius 1)
let dir_blending : f32 = 0.8; 
let wall_radius : f32 = 0.5;  

let friction : bool = true;

// ----- Struct -----
struct SimulationParams {
  deltaTime : f32;
  avoidance : f32;
  numAgents : f32;
  gridWidth : f32;
  iteration : i32;
  tick      : f32;
  farRadius : f32;
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

struct Agents {
  agents : array<Agent>;
};

struct CellIndices {
  start : u32;
  end   : u32;
};

struct Grid {
  cells : array<CellIndices>;
};

struct Obstacle {
  pos : vec3<f32>;
  rot : f32;
  scale : vec3<f32>;
};

struct Obstacles {
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


// --- Neighbor finding helper functions ---

fn cell2dto1d(x: i32, y: i32, gridWidth: f32) -> i32 {
  return x + (y * i32(gridWidth));
}

fn cell1dto2d(i: i32, gridWidth: f32) -> vec2<i32> {
  let x = i % i32(gridWidth);
  let y = i / i32(gridWidth);

  return vec2<i32>(x, y);
}

fn worldSpacePosToCellSpace(x: f32, 
                            z: f32, 
                            gridWidth: f32, 
                            cellWidth: f32) -> vec2<f32> {
  let pos = vec2<f32>(x + (cellWidth * gridWidth / 2.0), 
                      z + (cellWidth * gridWidth / 2.0));
  
  return pos;
}

fn cellSpaceToCell2d(x: f32, y: f32, cellWidth: f32) -> vec2<i32>{
  return vec2<i32>(i32(x / cellWidth), i32(y / cellWidth));
}


fn worldSpacePosToCell2d(x: f32, z: f32, gridWidth: f32, cellWidth: f32) -> vec2<i32> {
  let pos = worldSpacePosToCellSpace(x, z, gridWidth, cellWidth); 
  
  return cellSpaceToCell2d(pos.x, pos.y, cellWidth);
}

fn getBBoxCornerCells(worldX: f32, 
                      worldZ: f32, 
                      gridWidth: f32, 
                      cellWidth: f32,
                      radius: f32) -> vec4<i32>{

  let upperLeft = worldSpacePosToCell2d(worldX + radius, 
                                 worldZ + radius, 
                                 gridWidth, 
                                 cellWidth);

  let backRight = worldSpacePosToCell2d(worldX - radius, 
                                 worldZ - radius, 
                                 gridWidth, 
                                 cellWidth);

  return vec4<i32>(backRight.x,
                   backRight.y,
                   upperLeft.x,
                   upperLeft.y); 
}

// --- misc. debris ----
fn rainbowCycle(t: f32) -> vec4<f32>{
  let x = t * 0.025;
  let twoPiOverThree = 3.14159 * 2.0 / 3.0;
  let r = pow(2.0, sin(x + twoPiOverThree)) * 0.5;
  let g = pow(2.0, sin(x - twoPiOverThree)) * 0.25;
  let b = pow(2.0, sin(x)) * 0.25;

  return vec4<f32>(r, g, b, 1.0);

}