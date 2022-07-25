////////////////////////////////////////////////////////////////////////////////
// Obstacles Rendering
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
@binding(0) @group(0) var<uniform> scene : Scene;

struct VertexInput {
  @location(0) position : vec3<f32>,  // obstacle position (world space)
  @location(1) rotation : f32,
  @location(2) scale    : vec3<f32>,
  @location(3) mesh_pos : vec4<f32>,  // mesh vertex position (model space)
  @location(4) mesh_uv  : vec2<f32>,  // mesh vertex uv
  @location(5) mesh_nor : vec4<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
  @location(1)       normal   : vec4<f32>,
  @location(2)       fragPos  : vec4<f32>,
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {

  var c = cos(in.rotation);
  var s = sin(in.rotation);
  var model = mat4x4<f32>();
  model[0] = vec4<f32>(in.scale.x*c, 0.0, -in.scale.x*s, 0.0);
  model[1] = vec4<f32>(0.0, in.scale.y, 0.0, 0.0);
  model[2] = vec4<f32>(in.scale.z*s, 0.0, in.scale.z*c, 0.0);
  model[3] = vec4<f32>(in.position, 1.0);

  var rot = mat4x4<f32>();
  rot[0] = vec4<f32>(c, 0.0, s, 0.0);
  rot[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  rot[2] = vec4<f32>(-s, 0.0, c, 0.0);
  rot[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

  var out : VertexOutput;
  out.fragPos = model * in.mesh_pos;
  out.position = scene.cameraViewProjMatrix * out.fragPos;
  out.color = vec4<f32>(0.1);
  out.normal = normalize(rot * in.mesh_nor);
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  var lightDir = normalize(scene.lightPos - in.fragPos.xyz);
  var lambertTerm = dot(normalize(lightDir), normalize(in.normal.xyz));
  var lightColor = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  return vec4<f32>(0.8, 0.8, 0.8, 1.0) + lightColor * lambertTerm * 0.2;
}
