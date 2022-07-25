////////////////////////////////////////////////////////////////////////////////
// Crowd Render Pipeline
////////////////////////////////////////////////////////////////////////////////

@binding(0) @group(0) var<uniform> scene : Scene;
@binding(0) @group(1) var<uniform> model : Model;

struct VertexInput {
  @location(0) position : vec3<f32>,  // agent position (world space)
  @location(1) color    : vec4<f32>,  // agent color
  @location(2) velocity : vec3<f32>,  // agent velocity
  @location(3) mesh_pos : vec4<f32>,  // mesh vertex position (model space)
  @location(4) mesh_uv  : vec2<f32>,  // mesh vertex uv
  @location(5) mesh_nor : vec4<f32>,  // mesh vertex normal
  @location(6) mesh_col : vec3<f32>,  // mesh vertex color
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
  @location(1)       mesh_pos : vec4<f32>,
  @location(2)       mesh_uv  : vec2<f32>,
  @location(3)       mesh_nor : vec4<f32>,
  @location(4)       mesh_col : vec3<f32>,
  @location(5)       shadowPos : vec3<f32>,
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {

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

  var out : VertexOutput;
  out.mesh_pos = instance * model.modelMatrix * in.mesh_pos;
  out.position = scene.cameraViewProjMatrix * out.mesh_pos;
  out.color = in.color;
  out.mesh_uv = in.mesh_uv;

  out.mesh_nor = instance * model.modelMatrix * in.mesh_nor;
  out.mesh_col = in.mesh_col;

  if(scene.shadowOn > 0.99) {
    // Shadow Mapping
    var posFromLight : vec4<f32> = scene.lightViewProjMatrix * out.mesh_pos;

    out.shadowPos = vec3<f32>(
      posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5),
      posFromLight.z
    );
  }

  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@group(0) @binding(1) var shadowSampler: sampler_comparison;
@group(0) @binding(2) var shadowMap: texture_depth_2d;

const ambientFactor : f32 = 0.2;

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {

  var visibility : f32 = 1.0;
  if(scene.shadowOn > 0.99) {
    visibility = 0.0;
    for (var y : i32 = -1 ; y <= 1 ; y = y + 1) {
        for (var x : i32 = -1 ; x <= 1 ; x = x + 1) {
          var offset : vec2<f32> = vec2<f32>(
            f32(x) * 0.00048828,
            f32(y) * 0.00048828);

            visibility = visibility + textureSampleCompare(
            shadowMap, shadowSampler,
            in.shadowPos.xy + offset, in.shadowPos.z - 0.007);
        }
    }
    visibility = visibility / 9.0;
  }

  var lightDir = normalize(scene.lightPos - in.mesh_pos.xyz);
  var lambertTerm = max(dot(lightDir, normalize(in.mesh_nor.xyz)), 0.0);
  var lightingTerm = min(ambientFactor + visibility * lambertTerm, 1.0);

  var meshCol = vec4<f32>(in.mesh_col, 1.0);
  if (meshCol.r > 0.99 && meshCol.g > 0.99 && meshCol.b > 0.99){
    meshCol = in.color;
  }
  var albedo = in.color * 0.3 + meshCol * 0.7;
  return albedo + lightingTerm * vec4<f32>(0.5);
}
