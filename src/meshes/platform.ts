export const platformVertexSize = 4 * 14; // Byte size of one cube vertex.
export const platformPositionOffset = 0;
export const platformColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
export const platformUVOffset = 4 * 8;
export const platformNorOffset = 4 * 10;
export const platformVertexCount = 36;

let platformHeight = 20.0;
platformHeight -= 2.0;

// prettier-ignore
export const platformVertexArray = new Float32Array([
  // float4 position, float4 color, float2 uv, float4 normal
  1, -platformHeight, 1, 1,   1, 1, 1, 1,  -1, -1,  0, -1, 0, 0,
  -1, -platformHeight, 1, 1,  1, 1, 1, 1,  -1, -1,  0, -1, 0, 0,
  -1, -platformHeight, -1, 1, 1, 1, 1, 1,  -1, -1,  0, -1, 0, 0,
  1, -platformHeight, -1, 1,  1, 0, 0, 1,  -1, -1,  0, -1, 0, 0,
  1, -platformHeight, 1, 1,   1, 0, 1, 1,  -1, -1,  0, -1, 0, 0,
  -1, -platformHeight, -1, 1, 0, 0, 0, 1,  -1, -1,  0, -1, 0, 0,

  1, 2, 1, 1,                 1, 1, 1, 1,  0, 0,  0, 0, -1, 0,
  1, -platformHeight, 1, 1,   1, 0, 1, 1,  0, 0,  0, 0, -1, 0,
  1, -platformHeight, -1, 1,  1, 0, 0, 1,  0, 0,  0, 0, -1, 0,
  1, 2, -1, 1,                1, 1, 0, 1,  0, 0,  0, 0, -1, 0,
  1, 2, 1, 1,                 1, 1, 1, 1,  0, 0,  0, 0, -1, 0,
  1, -platformHeight, -1, 1,  1, 0, 0, 1,  0, 0,  0, 0, -1, 0,

  -1, 2, 1, 1,                0, 1, 1, 1,  1, 1,  0, 1, 0, 0,
  1, 2, 1, 1,                 1, 1, 1, 1,  0, 1,  0, 1, 0, 0,
  1, 2, -1, 1,                1, 1, 0, 1,  0, 0,  0, 1, 0, 0,
  -1, 2, -1, 1,               0, 1, 0, 1,  1, 0,  0, 1, 0, 0,
  -1, 2, 1, 1,                0, 1, 1, 1,  1, 1,  0, 1, 0, 0,
  1, 2, -1, 1,                1, 1, 0, 1,  0, 0,  0, 1, 0, 0,

  -1, -platformHeight, 1, 1,  0, 0, 1, 1,  0, 0,  0, 0, 1, 0,
  -1, 2, 1, 1,                0, 1, 1, 1,  0, 0,  0, 0, 1, 0,
  -1, 2, -1, 1,               0, 1, 0, 1,  0, 0,  0, 0, 1, 0,
  -1, -platformHeight, -1, 1, 0, 0, 0, 1,  0, 0,  0, 0, 1, 0,
  -1, -platformHeight, 1, 1,  0, 0, 1, 1,  0, 0,  0, 0, 1, 0,
  -1, 2, -1, 1,               0, 1, 0, 1,  0, 0,  0, 0, 1, 0,

  1, 2, 1, 1,                 1, 1, 1, 1,  0, 0,  -1, 0, 0, 0,
  -1, 2, 1, 1,                0, 1, 1, 1,  0, 0,  -1, 0, 0, 0,
  -1, -platformHeight, 1, 1,  0, 0, 1, 1,  0, 0,  -1, 0, 0, 0,
  -1, -platformHeight, 1, 1,  0, 0, 1, 1,  0, 0,  -1, 0, 0, 0,
  1, -platformHeight, 1, 1,   1, 0, 1, 1,  0, 0,  -1, 0, 0, 0,
  1, 2, 1, 1,                 1, 1, 1, 1,  0, 0,  -1, 0, 0, 0,

  1, -platformHeight, -1, 1,  1, 0, 0, 1,  0, 0,  1, 0, 0, 0,
  -1, -platformHeight, -1, 1, 0, 0, 0, 1,  0, 0,  1, 0, 0, 0,
  -1, 2, -1, 1,               0, 1, 0, 1,  0, 0,  1, 0, 0, 0,
  1, 2, -1, 1,                1, 1, 0, 1,  0, 0,  1, 0, 0, 0,
  1, -platformHeight, -1, 1,  1, 0, 0, 1,  0, 0,  1, 0, 0, 0,
  -1, 2, -1, 1,               0, 1, 0, 1,  0, 0,  1, 0, 0, 0
]);