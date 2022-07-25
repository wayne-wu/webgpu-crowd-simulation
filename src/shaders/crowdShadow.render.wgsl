////////////////////////////////////////////////////////////////////////////////
// Crowd Shadow Rendering Pipeline
// Based on : https://austin-eng.com/webgpu-samples/samples/shadowMapping
////////////////////////////////////////////////////////////////////////////////

struct VertexInput {
  @location(0) position : vec3<f32>,  // agent position (world space)
  @location(1) velocity : vec3<f32>,  // agent velocity
  @location(2) mesh_pos : vec4<f32>,  // mesh vertex position (model space)
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn vs_main(in : VertexInput)
     -> @builtin(position) vec4<f32> {

  var vel = normalize(in.velocity);

  var instance = mat4x4<f32>();
  var scale = mat4x4<f32>();
  scale[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  scale[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  scale[2] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  scale[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var rot = mat4x4<f32>();
  var forward = vel;
  var up = vec3<f32>(0.0, 1.0, 0.0);
  var right = normalize(cross(forward, up));
  rot[0] = vec4<f32>(right, 0.0);
  rot[1] = vec4<f32>(up, 0.0);
  rot[2] = vec4<f32>(forward, 0.0);
  rot[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var trans = mat4x4<f32>();
  trans[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  trans[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  trans[2] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  trans[3] = vec4<f32>(in.position, 1.0);

  instance = trans * rot * scale;

  return scene.lightViewProjMatrix * instance * model.modelMatrix * vec4<f32>(in.mesh_pos.xyz, 1.0);
}

// ONLY USES VERTEX SHADER
@fragment
fn fs_main() {
}