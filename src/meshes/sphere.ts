import * as THREE from 'three';

// radius, horizontal segments, vertical segments
const sphere = new THREE.SphereGeometry( 1, 16, 16 );

const spherePosArray = sphere.attributes.position.array;
const sphereNorArray = sphere.attributes.normal.array;
export const sphereVertCount = sphere.index.count;

export const sphereVertexArray = [];
export const spherePosOffset = 0;
export const sphereNorOffset = 4 * 4;
export const sphereItemSize = 4 * 4 + // position
                              4 * 4;  // normal
const idxArray = sphere.index.array;

for (let i = 0; i < idxArray.length; i++) {
    const idx = idxArray[i];
    sphereVertexArray[i * 8 + 0] = spherePosArray[idx * 3 + 0];
    sphereVertexArray[i * 8 + 1] = spherePosArray[idx * 3 + 1];
    sphereVertexArray[i * 8 + 2] = spherePosArray[idx * 3 + 2];
    sphereVertexArray[i * 8 + 3] = 1.0;
    sphereVertexArray[i * 8 + 4] = sphereNorArray[idx * 3 + 0];
    sphereVertexArray[i * 8 + 5] = sphereNorArray[idx * 3 + 1];
    sphereVertexArray[i * 8 + 6] = sphereNorArray[idx * 3 + 2];
    sphereVertexArray[i * 8 + 7] = 0.0;
}