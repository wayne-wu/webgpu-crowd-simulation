export const cubeVertexSize = 4 * 13; // Byte size of one cube vertex.
export const cubePositionOffset = 0;
export const cubeColorOffset = 4 * 10; // Byte offset of cube vertex color attribute.
export const cubeUVOffset = 4 * 4;
export const cubeNorOffset = 4 * 6;
export const cubeVertexCount = 36;

// prettier-ignore
export const cubeVertexArray = new Float32Array([
  // float4 position, float2 uv, float4 normal, float3 color, 
  // has to follow mesh order, because it is loaded in like a mesh
  1, -1, 1, 1,   1, 1,  0, -1, 0, 0,  1, 1, 1,
  -1, -1, 1, 1,  0, 1,  0, -1, 0, 0,  1, 1, 1,
  -1, -1, -1, 1, 0, 0,  0, -1, 0, 0,  1, 1, 1,
  1, -1, -1, 1,  1, 0,  0, -1, 0, 0,  1, 1, 1,
  1, -1, 1, 1,   1, 1,  0, -1, 0, 0,  1, 1, 1,
  -1, -1, -1, 1, 0, 0,  0, -1, 0, 0,  1, 1, 1,

  1, 1, 1, 1,    1, 1,  0, 0, -1, 0,  1, 1, 1,
  1, -1, 1, 1,   0, 1,  0, 0, -1, 0,  1, 1, 1,
  1, -1, -1, 1,  0, 0,  0, 0, -1, 0,  1, 1, 1,
  1, 1, -1, 1,   1, 0,  0, 0, -1, 0,  1, 1, 1,
  1, 1, 1, 1,    1, 1,  0, 0, -1, 0,  1, 1, 1,
  1, -1, -1, 1,  0, 0,  0, 0, -1, 0,  1, 1, 1,

  -1, 1, 1, 1,   1, 1,  0, 1, 0, 0,  1, 1, 1,
  1, 1, 1, 1,    0, 1,  0, 1, 0, 0,  1, 1, 1,
  1, 1, -1, 1,   0, 0,  0, 1, 0, 0,  1, 1, 1,
  -1, 1, -1, 1,  1, 0,  0, 1, 0, 0,  1, 1, 1,
  -1, 1, 1, 1,   1, 1,  0, 1, 0, 0,  1, 1, 1,
  1, 1, -1, 1,   0, 0,  0, 1, 0, 0,  1, 1, 1,

  -1, -1, 1, 1,  1, 1,  0, 0, 1, 0,  1, 1, 1,
  -1, 1, 1, 1,   0, 1,  0, 0, 1, 0,  1, 1, 1,
  -1, 1, -1, 1,  0, 0,  0, 0, 1, 0,  1, 1, 1,
  -1, -1, -1, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1,
  -1, -1, 1, 1,  1, 1,  0, 0, 1, 0,  1, 1, 1,
  -1, 1, -1, 1,  0, 0,  0, 0, 1, 0,  1, 1, 1,

  1, 1, 1, 1,    1, 1,  -1, 0, 0, 0,  1, 1, 1,
  -1, 1, 1, 1,   0, 1,  -1, 0, 0, 0,  1, 1, 1,
  -1, -1, 1, 1,  0, 0,  -1, 0, 0, 0,  1, 1, 1,
  -1, -1, 1, 1,  0, 0,  -1, 0, 0, 0,  1, 1, 1,
  1, -1, 1, 1,   1, 0,  -1, 0, 0, 0,  1, 1, 1,
  1, 1, 1, 1,    1, 1,  -1, 0, 0, 0,  1, 1, 1,

  1, -1, -1, 1,  1, 1,  1, 0, 0, 0,  1, 1, 1,
  -1, -1, -1, 1, 0, 1,  1, 0, 0, 0,  1, 1, 1,
  -1, 1, -1, 1,  0, 0,  1, 0, 0, 0,  1, 1, 1,
  1, 1, -1, 1,   1, 0,  1, 0, 0, 0,  1, 1, 1,
  1, -1, -1, 1,  1, 1,  1, 0, 0, 0,  1, 1, 1,
  -1, 1, -1, 1,  0, 0,  1, 0, 0, 0,  1, 1, 1,
]);