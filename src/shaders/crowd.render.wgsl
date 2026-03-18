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
  @location(4) mesh_nor : vec4<f32>,  // mesh vertex normal
  @location(5) right    : vec3<f32>,  // agent right dir
  @location(6) cell     : i32,        // agent spatial hash cell
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0)       color    : vec4<f32>,
  @location(1)       mesh_pos : vec4<f32>,
  @location(2)       mesh_nor : vec4<f32>,
  @location(3)       shadowPos : vec3<f32>,
  @location(4) @interpolate(flat) cell : i32,
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  let modelPos = model.modelMatrix * in.mesh_pos;
  let modelNor = model.modelMatrix * in.mesh_nor;
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let worldPos = in.position +
    modelPos.x * in.right +
    modelPos.y * up +
    modelPos.z * in.velocity;
  let worldNor = modelNor.x * in.right +
    modelNor.y * up +
    modelNor.z * in.velocity;

  var out : VertexOutput;
  out.mesh_pos = vec4<f32>(worldPos, modelPos.w);
  out.position = scene.cameraViewProjMatrix * out.mesh_pos;
  out.color = in.color;

  out.mesh_nor = vec4<f32>(worldNor, modelNor.w);
  out.cell = in.cell;

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

fn hash_color(cell: i32) -> vec4<f32> {
  let x = f32(cell);
  return vec4<f32>(
    0.5 + 0.5 * sin(0.13 * x + 0.0),
    0.5 + 0.5 * sin(0.17 * x + 2.0),
    0.5 + 0.5 * sin(0.19 * x + 4.0),
    1.0
  );
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

  var baseColor = in.color;
  if (scene.debugCell > 0.99) {
    baseColor = hash_color(in.cell);
  }

  var albedo = baseColor;
  return albedo + lightingTerm * vec4<f32>(0.5);
}
