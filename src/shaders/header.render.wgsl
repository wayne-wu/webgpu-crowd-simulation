////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////
var<private> rand_seed : vec2<f32>;

fn rand() -> f32 {
    rand_seed.x = fract(cos(dot(rand_seed, vec2<f32>(23.14077926, 232.61690225))) * 136.8168);
    rand_seed.y = fract(cos(dot(rand_seed, vec2<f32>(54.47856553, 345.84153136))) * 534.7645);
    return rand_seed.y;
}

struct Scene {
    lightViewProjMatrix : mat4x4<f32>;
    cameraViewProjMatrix : mat4x4<f32>;
    lightPos : vec3<f32>;
    gridOn : f32;
    cameraPos : vec3<f32>;
    time : f32;
    shadowOn : f32;
};

struct Model {
    modelMatrix : mat4x4<f32>;
};

