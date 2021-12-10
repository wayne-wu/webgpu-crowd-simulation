////////////////////////////////////////////////////////////////////////////////
// Finalize Velocity Compute Shader
////////////////////////////////////////////////////////////////////////////////

[[binding(0), group(0)]] var<uniform> sim_params : SimulationParams;
[[binding(1), group(0)]] var<storage, read> agentData_r : Agents;
[[binding(2), group(0)]] var<storage, write> agentData_w : Agents;
[[binding(3), group(0)]] var<storage, read> grid : Grid;
[[binding(4), group(0)]] var<storage, read> obstacleData : Obstacles;


fn intersect_line(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, n: ptr<function, vec2<f32>>) -> f32
{
  let s1 = p1 - p0;
  let s2 = p3 - p2;

  var den = (-s2.x * s1.y + s1.x * s2.y);
  if (den < eps) { return -1.0; }  // colinear

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

fn obstacle_avoidance(agent: Agent, obstacle: Obstacle) -> vec3<f32>
{
  // Create Model Matrix
  let c = cos(obstacle.rot);
  let s = sin(obstacle.rot);
  var m = mat4x4<f32>();
  m[0] = vec4<f32>(obstacle.scale.x*c, 0.0, -obstacle.scale.x*s, 0.0);
  m[1] = vec4<f32>(0.0, obstacle.scale.y, 0.0, 0.0);
  m[2] = vec4<f32>(obstacle.scale.z*s, 0.0, obstacle.scale.z*c, 0.0);
  m[3] = vec4<f32>(obstacle.pos, 1.0);

  // Get Corner Points in World Position (Cube)
  let l = 1.05;
  var p1 = (m * vec4<f32>(l,0.0,l,1.0)).xz;
  var p2 = (m * vec4<f32>(l,0.0,-l,1.0)).xz;
  var p3 = (m * vec4<f32>(-l,0.0,-l,1.0)).xz;
  var p4 = (m * vec4<f32>(-l,0.0,l,1.0)).xz;

  var v = agent.v;
  var a0 = agent.xp.xz;
  var a1 = (agent.xp + tObstacle * v).xz;  // max look-ahead
  
  // Intersection test with the four edges
  var n_tmp : vec2<f32>;
  var n_min : vec2<f32>;
  var t_tmp : f32;
  var t_min : f32 = tObstacle;
  t_tmp = intersect_line(a0, a1, p1, p2, &n_tmp);
  if (t_tmp > eps && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p2, p3, &n_tmp);
  if (t_tmp > eps && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p3, p4, &n_tmp);
  if (t_tmp > eps && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }
  t_tmp = intersect_line(a0, a1, p4, p1, &n_tmp);
  if (t_tmp > eps && t_tmp < t_min) { t_min = t_tmp; n_min = n_tmp; }

  if (t_min < 1.0) { 
    t_min = t_min * tObstacle;  // remap t_min to 0 to tObstacle
    
    //if(dot(v.xz, n_min) > 0.0) { n_min = -n_min; }  // flip the normal direction
    //var n = vec3<f32>(n_min.x, 0.0, n_min.y);  // contact normal

    // Use the radial normal as the contact normal so that there's some tangential velocity
    var n = normalize((agent.xp + t_min * v) - obstacle.pos);
  
    return k_avoid * n;
  }

  return vec3<f32>(0.0);
}

fn getW(r : f32) -> f32 {
    var w = 0.0; // poly6 smoothing kernel

    if (eps <= r && r <= xsph_h) {
        w = 315.0 / (64.0 * 3.14159 * pow(xsph_h, 9.0));
        let hmr = xsph_h * xsph_h - r * r;
        w = w * hmr * hmr * hmr;
    }
    return w;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
  let idx = GlobalInvocationID.x;
  var agent = agentData_r.agents[idx];

  // PBD: Get new velocity from corrected position
  agent.v = (agent.xp - agent.x)/sim_params.deltaTime;

  // 4.3 Cohesion
  // update velocity to factor in viscosity
  var velAvg = vec3<f32>(0.0); // weighted average of all the velocity differences

  if (agent.cell < 0){
    // ignore invalid cells
    agent.c = vec4<f32>(0.5, 0.5, 0.5, 1.0);
    agentData_w.agents[idx] = agent;
    return;
  }

  let gridWidth = sim_params.gridWidth;
  let gridHeight = gridWidth;//sim_params.gridWidth;
  // TODO don't hardcode
  let cellWidth = 1000.0 / gridWidth;
  // compute cells that could conceivably contain neighbors
  let bboxCorners = getBBoxCornerCells(agent.x.x,
                                       agent.x.z,
                                       gridWidth,
                                       cellWidth,
                                       nearRadius);

  let minX = bboxCorners[0];
  let minY = bboxCorners[1];
  let maxX = bboxCorners[2];
  let maxY = bboxCorners[3];

  //for (var c : u32 = 0u; c < cellsToCheck; c = c + 1u ){
  for (var cellY = minY; cellY <= maxY; cellY = cellY + 1){
    if (cellY < 0 || cellY >= i32(gridHeight)){
      continue;
    }
    for (var cellX = minX; cellX <= maxX; cellX = cellX + 1){

      if (cellX < 0 || cellX >= i32(gridWidth)){
        continue;
      }
      let cellIdx = cell2dto1d(cellX, cellY, gridWidth);
      let cell : CellIndices = grid.cells[cellIdx];
      for (var i : u32 = cell.start; i <= cell.end; i = i + 1u) {

        if (idx == i) { 
          // ignore ourselves
          continue; 
        }

        var neighbor = agentData_r.agents[i];
        
        // Only look at neighbor in the same crowd group
        if(agent.group != neighbor.group){
          continue;
        }
        
        var d = distance(agent.xp, neighbor.xp);  // Should this be xp or x?
        if (d >= nearRadius){
          continue;
        }
        var w = getW(d*d);
        velAvg = velAvg + (agent.v - neighbor.v) * w;
      }
    }
  }
  agent.v = agent.v + xsph_c * velAvg;

  // 4.7 Obstacle Avoidance (Open Steer)
  for (var j : u32 = 0u; j < arrayLength(&obstacleData.obstacles); j = j + 1u){
    var v_avoid = obstacle_avoidance(agent, obstacleData.obstacles[j]);
    agent.v = agent.v + v_avoid;
  }

  // 4.6 Maximum Speed and Acceleration Limiting

  let v_dir = normalize(agent.v);
  let maxSpeed : f32 = 1.2*agent.speed;
  if(length(agent.v) > maxSpeed){
    agent.v = maxSpeed * v_dir;
  }

  // Set new position to be the corrected position
  // Reintegrate here so that the position doesn't jump between frames
  agent.x = agent.x + agent.v * sim_params.deltaTime;

  agent.dir = dir_blending * normalize(agent.dir) + (1.0 - dir_blending) * v_dir;

  // Store the new agent value
  agentData_w.agents[idx] = agent;
}
