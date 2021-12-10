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
//            Fragment Shader for GridLines                         //
////////////////////////////////////////////////////////////////////// 

// [[stage(fragment)]]
// fn fs_gridLines([[location(0)]] fragUV: vec2<f32>,
//         [[location(1)]] fragPosition: vec4<f32>) -> [[location(0)]] vec4<f32> {
//   return vec4<f32>(0.0, 0.0, 0.0, 1.0);
// }

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
    albedo = textureSample(myTexture, mySampler, in.fragUV) + 0.95;
  }

  return 0.2*albedo + 0.8*vec4<f32>(lightingTerm * albedo.xyz, 1.0);;
}
