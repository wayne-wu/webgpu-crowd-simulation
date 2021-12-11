//////////////////////////////////////////////////////////////////////
//                Platform Render Pipeline                          //
////////////////////////////////////////////////////////////////////// 


[[binding(0), group(0)]] var<uniform> scene : Scene;
[[binding(0), group(1)]] var<uniform> model : Model;
  
struct VertexOutput {
  [[builtin(position)]] position : vec4<f32>;
  [[location(0)]] fragUV : vec2<f32>;
  [[location(1)]] fragPos: vec4<f32>;
  [[location(2)]] fragNor : vec3<f32>;
  [[location(3)]] shadowPos : vec3<f32>;
};
  
[[stage(vertex)]]
fn vs_main([[location(0)]] position : vec4<f32>,
            [[location(1)]] uv : vec2<f32>,
            [[location(2)]] nor : vec4<f32>) -> VertexOutput {
  var output : VertexOutput;
  output.fragPos = model.modelMatrix * position;
  output.position = scene.cameraViewProjMatrix * output.fragPos;
  output.fragUV = uv;
  output.fragNor = nor.xyz;

  // Calculate shadow pos here to be interpolated in fs
  // XY is in (-1, 1) space, Z is in (0, 1) space
  let posFromLight : vec4<f32> = scene.lightViewProjMatrix * output.fragPos;

  // Convert XY to (0, 1)
  // Y is flipped because texture coords are Y-down.
  output.shadowPos = vec3<f32>(
    posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5),
    posFromLight.z
  );

  return output;
}

//////////////////////////////////////////////////////////////////////
//            Fragment Shader for Platform                          //
////////////////////////////////////////////////////////////////////// 

[[group(0), binding(1)]] var mySampler: sampler;
[[group(0), binding(2)]] var myTexture: texture_2d<f32>;
[[group(0), binding(3)]] var shadowSampler: sampler_comparison;
[[group(0), binding(4)]] var shadowMap: texture_depth_2d;

let ambientFactor = 0.2;

[[stage(fragment)]]
fn fs_platform(in : VertexOutput) -> [[location(0)]] vec4<f32> {

  var visibility : f32 = 0.0;
  for (var y : i32 = -1 ; y <= 1 ; y = y + 1) {
      for (var x : i32 = -1 ; x <= 1 ; x = x + 1) {
        // NOTE: Must change the texel offset size if texture size is changed
        let offset : vec2<f32> = vec2<f32>(
          f32(x) * 0.00048828,
          f32(y) * 0.00048828);

          visibility = visibility + textureSampleCompare(
          shadowMap, shadowSampler,
          in.shadowPos.xy + offset, in.shadowPos.z - 0.007);
      }
  }
  visibility = visibility / 9.0;

  var lightDir = normalize(scene.lightPos - in.fragPos.xyz);
  var lambertTerm = dot(normalize(lightDir), normalize(in.fragNor));
  let lightingTerm = min(ambientFactor + visibility * lambertTerm, 1.0);

  if (in.fragUV.x == -1.0){
    return vec4<f32>(0.9, 0.9, 0.9, 1.0);
  }
  var albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  
  if (scene.gridOn > 0.99){
    albedo = textureSample(myTexture, mySampler, in.fragUV) + 0.98;
  }

  return 0.2*albedo + 0.8*vec4<f32>(lightingTerm * albedo.xyz, 1.0);;
}

////////////////////////////////////////////////////////////////////////
//                       Shaders for Goals                            //
////////////////////////////////////////////////////////////////////////

struct VertexOutputGoal {
    [[builtin(position)]] Position : vec4<f32>;
    [[location(0)]] fragPosition: vec4<f32>;
    [[location(1)]] fragNor : vec4<f32>;
};

[[stage(vertex)]]
  fn vs_goal([[location(0)]] goalPos : vec4<f32>,
             [[location(1)]] meshPos : vec4<f32>,
             [[location(2)]] meshNor : vec4<f32>) -> VertexOutputGoal {

    // Scale the goal based on distance to camera
    var s = 0.01*distance(scene.cameraPos, goalPos.xyz);

    var model = mat4x4<f32>();
    model[0] = vec4<f32>(s, 0.0, 0.0, 0.0);
    model[1] = vec4<f32>(0.0, s, 0.0, 0.0);
    model[2] = vec4<f32>(0.0, 0.0, s, 0.0);
    model[3] = vec4<f32>(goalPos[0], goalPos[1], goalPos[2], 1.0);

    var output : VertexOutputGoal;
    output.Position = scene.cameraViewProjMatrix * model * meshPos;
    output.fragPosition = meshPos;
    output.fragNor = meshNor;
    return output;
  }

[[stage(fragment)]]
fn fs_goal([[location(0)]] fragPosition: vec4<f32>,
               [[location(1)]] fragNor : vec4<f32>) -> [[location(0)]] vec4<f32> {
  var lightDir = vec4<f32>(1.0, 1.0, 1.0, 0.0);
  var lambertTerm = dot(normalize(lightDir), normalize(fragNor));

  var ambient = vec4<f32>(0.2, 0.5, 0.2, 0.0);
  var albedo = vec4<f32>(0.0, 1.0, 0.0, 1.0);

  // move color palette along sphere in y direction
  var y = (fragPosition.y + 1.0) / 2.0;
  y = fract(y + scene.time * 0.005);

  // cosine color palette
  var a = vec3<f32>(0.608, 0.718, 0.948);
  var b = vec3<f32>(0.858, 0.248, 0.308);
  var c = vec3<f32>(-1.112, 1.000, 1.000);
  var d = vec3<f32>(0.000, 0.333, 0.667);
  albedo = vec4<f32>(a + b*cos( 6.28318*(c*y+d) ), 1.0);

  return albedo + ambient * lambertTerm;
}
