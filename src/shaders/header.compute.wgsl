// ----- Parameteres -----
const eps : f32 = 0.0001;
const t0 : f32 = 20.0;            // paper = 20
const tObstacle : f32 = 20.0;
const k_longrange : f32 = 0.15;   // paper = 0.24 [0-1]
const k_obstacle : f32 = 0.24;
const k_shortrange : f32 = 1.0;   // paper = 1.0 [0,1]
const k_avoid : f32 = 0.2;
const avgCoefficient : f32 = 1.2; // paper = 1.2 [1,2]
// const farRadius : f32 = 6.0;   // Promoted to GUI
const nearRadius : f32 = 2.0;
const cohesionRadius : f32 = 5.0;
const mu_static : f32 = 0.21;     // paper = 0.21
const mu_kinematic : f32 = 0.10;  // papser = ?
const ksi : f32 = 0.0385;         // paper = 0.0385
const xsph_c : f32 = 7.0;         // paper = 7.0
const xsph_h : f32 = 217.0;       // paper = 217.0 // the smoothing distance specified in the paper (assumes particles with radius 1)
const dir_blending : f32 = 0.8; 
const wall_radius : f32 = 0.5;  

const friction : bool = true;

// ----- Struct -----
struct SimulationParams {
  deltaTime : f32,
  avoidance : f32,
  numAgents : f32,
  gridWidth : f32,
  iteration : i32,
  tick      : f32,
  farRadius : f32,
}

struct Agent {
  x  : vec3<f32>,  // position + radius
  r  : f32,
  c  : vec4<f32>,  // color
  v  : vec3<f32>,  // velocity + inverse mass
  w  : f32,
  xp : vec3<f32>,  // planned/predicted position
  speed : f32,
  goal : vec3<f32>,
  cell : i32,
  dir : vec3<f32>,
  group : f32,
}

struct Agents {
  agents : array<Agent>,
}

struct CellIndices {
  start : u32,
  end   : u32,
}

struct Grid {
  cells : array<CellIndices>,
}

struct Obstacle {
  pos : vec3<f32>,
  rot : f32,
  scale : vec3<f32>,
}

struct Obstacles {
  obstacles : array<Obstacle>,
}


fn wall_constraint(agent: Agent, p0: vec2<f32>, p1: vec2<f32>, count: ptr<function, i32>, total_dx: ptr<function, vec3<f32>>) -> bool{
  var xp = agent.xp.xz;
  var a = xp - p0;
  var b = p1 - p0;
  var b_norm = normalize(b);
  var l = dot(a, b_norm);

  if(l < eps || l > length(b)) { return false; }

  var c = l * b_norm;

  var xj = p0 + c; 
  var n = a - c;
  var d = length(n);

  var r = agent.r + wall_radius;
  var f = d - r;
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
  var c = cos(obstacle.rot);
  var s = sin(obstacle.rot);
  var m = mat4x4<f32>();
  m[0] = vec4<f32>(obstacle.scale.x*c, 0.0, -obstacle.scale.x*s, 0.0);
  m[1] = vec4<f32>(0.0, obstacle.scale.y, 0.0, 0.0);
  m[2] = vec4<f32>(obstacle.scale.z*s, 0.0, obstacle.scale.z*c, 0.0);
  m[3] = vec4<f32>(obstacle.pos, 1.0);

  // Get Corner Points in World Position (Cube)
  const l = 1.0;
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
  var x = i % i32(gridWidth);
  var y = i / i32(gridWidth);

  return vec2<i32>(x, y);
}

fn worldSpacePosToCellSpace(x: f32, 
                            z: f32, 
                            gridWidth: f32, 
                            cellWidth: f32) -> vec2<f32> {
  var pos = vec2<f32>(x + (cellWidth * gridWidth / 2.0), 
                      z + (cellWidth * gridWidth / 2.0));
  
  return pos;
}

fn cellSpaceToCell2d(x: f32, y: f32, cellWidth: f32) -> vec2<i32>{
  return vec2<i32>(i32(x / cellWidth), i32(y / cellWidth));
}


fn worldSpacePosToCell2d(x: f32, z: f32, gridWidth: f32, cellWidth: f32) -> vec2<i32> {
  var pos = worldSpacePosToCellSpace(x, z, gridWidth, cellWidth); 
  
  return cellSpaceToCell2d(pos.x, pos.y, cellWidth);
}

fn getBBoxCornerCells(worldX: f32, 
                      worldZ: f32, 
                      gridWidth: f32, 
                      cellWidth: f32,
                      radius: f32) -> vec4<i32>{

  var upperLeft = worldSpacePosToCell2d(worldX + radius, 
                                 worldZ + radius, 
                                 gridWidth, 
                                 cellWidth);

  var backRight = worldSpacePosToCell2d(worldX - radius, 
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
  var x = t * 0.025;
  var twoPiOverThree = 3.14159 * 2.0 / 3.0;
  var r = pow(2.0, sin(x + twoPiOverThree)) * 0.5;
  var g = pow(2.0, sin(x - twoPiOverThree)) * 0.25;
  var b = pow(2.0, sin(x)) * 0.25;

  return vec4<f32>(r, g, b, 1.0);

}