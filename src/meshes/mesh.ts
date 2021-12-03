import { Scene} from 'three';
import {GLTFLoader} from '../../node_modules/three/examples/jsm/loaders/GLTFLoader.js';

//const GLTFLoader = require('three/examples/jsm/loaders/GLTFLoader.js');

export const meshVertexSize = 4 * 4; // Position
export const meshPositionOffset = 0;
//export const meshColorOffset = 4 * 4; // Byte offset of cube vertex color attribute.
//export const meshUVOffset = 4 * 8;
export let meshVertexCount = -1;

export const meshVertexArray = new Float32Array();

// function createModel() {
//     let GLTFLoader = require('three/examples/jsm/loaders/GLTFLoader').GLTFLoader;
 let scene = new Scene();

new GLTFLoader()
	.load( '../../public/char1.glb', function ( gltf ) {

		scene.add( gltf.scene );
		//const object = gltf.scene.getObjectByName( 'char1_fabric' );

        gltf.scene.traverse( function ( child ) {
            if ( child.isMesh ) {
                meshVertexCount = child.geometry.attributes.position.count;
                console.log(child.geometry.attributes.position.count);    
            }
        })

	} );
//}

//createModel();

// import React, { Suspense, useEffect, useRef } from 'react'
// import { Canvas, useLoader, useThree  } from 'react-three-fiber'

// let GLTFLoader;
// let test = 0;

// function Model(props) {
//   GLTFLoader = require('three/examples/jsm/loaders/GLTFLoader').GLTFLoader;
//   const group = useRef();
//   const gltf = useLoader(GLTFLoader, 'glb/model.glb');

//   gltf.scene.traverse( function ( child ) {
//                  if ( child.isMesh ) {
//                      test = child.geometry.attributes.position.count;
//                      console.log(child.geometry.attributes.position.count);    
//                  }
//              })
  
//   return ('');
// }

// import dragonRawData from 'stanford-dragon/4';
// import { vec3 } from 'gl-matrix';

// function computeSurfaceNormals(
//   positions: [number, number, number][],
//   triangles: [number, number, number][]
// ): [number, number, number][] {
//   const normals: [number, number, number][] = positions.map(() => {
//     // Initialize to zero.
//     return [0, 0, 0];
//   });
//   triangles.forEach(([i0, i1, i2]) => {
//     const p0 = positions[i0];
//     const p1 = positions[i1];
//     const p2 = positions[i2];

//     const v0 = vec3.subtract(vec3.create(), p1, p0);
//     const v1 = vec3.subtract(vec3.create(), p2, p0);

//     vec3.normalize(v0, v0);
//     vec3.normalize(v1, v1);
//     const norm = vec3.cross(vec3.create(), v0, v1);

//     // Accumulate the normals.
//     vec3.add(normals[i0], normals[i0], norm);
//     vec3.add(normals[i1], normals[i1], norm);
//     vec3.add(normals[i2], normals[i2], norm);
//   });
//   normals.forEach((n) => {
//     // Normalize accumulated normals.
//     vec3.normalize(n, n);
//   });

//   return normals;
// }

// type ProjectedPlane = 'xy' | 'xz' | 'yz';

// const projectedPlane2Ids: { [key in ProjectedPlane]: [number, number] } = {
//   xy: [0, 1],
//   xz: [0, 2],
//   yz: [1, 2],
// };

// function computeProjectedPlaneUVs(
//   positions: [number, number, number][],
//   projectedPlane: ProjectedPlane = 'xy'
// ): [number, number][] {
//   const idxs = projectedPlane2Ids[projectedPlane];
//   const uvs: [number, number][] = positions.map(() => {
//     // Initialize to zero.
//     return [0, 0];
//   });
//   const extentMin = [Infinity, Infinity];
//   const extentMax = [-Infinity, -Infinity];
//   positions.forEach((pos, i) => {
//     // Simply project to the selected plane
//     uvs[i][0] = pos[idxs[0]];
//     uvs[i][1] = pos[idxs[1]];

//     extentMin[0] = Math.min(pos[idxs[0]], extentMin[0]);
//     extentMin[1] = Math.min(pos[idxs[1]], extentMin[1]);
//     extentMax[0] = Math.max(pos[idxs[0]], extentMax[0]);
//     extentMax[1] = Math.max(pos[idxs[1]], extentMax[1]);
//   });
//   uvs.forEach((uv) => {
//     uv[0] = (uv[0] - extentMin[0]) / (extentMax[0] - extentMin[0]);
//     uv[1] = (uv[1] - extentMin[1]) / (extentMax[1] - extentMin[1]);
//   });
//   return uvs;
// }

// export const mesh = {
//   positions: dragonRawData.positions as [number, number, number][],
//   triangles: dragonRawData.cells as [number, number, number][],
//   normals: [] as [number, number, number][],
//   uvs: [] as [number, number][],
// };

// // Compute surface normals
// mesh.normals = computeSurfaceNormals(mesh.positions, mesh.triangles);

// // Compute some easy uvs for testing
// mesh.uvs = computeProjectedPlaneUVs(mesh.positions, 'xy');

// // Push indices for an additional ground plane
// mesh.triangles.push(
//   [mesh.positions.length, mesh.positions.length + 2, mesh.positions.length + 1],
//   [mesh.positions.length, mesh.positions.length + 1, mesh.positions.length + 3]
// );

// // Push vertex attributes for an additional ground plane
// // prettier-ignore
// mesh.positions.push(
//   [-100, 20, -100], //
//   [ 100, 20,  100], //
//   [-100, 20,  100], //
//   [ 100, 20, -100]
// );
// mesh.normals.push(
//   [0, 1, 0], //
//   [0, 1, 0], //
//   [0, 1, 0], //
//   [0, 1, 0]
// );
// mesh.uvs.push(
//   [0, 0], //
//   [1, 1], //
//   [0, 1], //
//   [1, 0]
// );

// export let meshVertexArray = new Float32Array();
// export let meshVertexCount = mesh.positions.length;
// let idx = 0;

// for (let i = 0; i < mesh.positions.length; i++) {
//     meshVertexArray[idx] = mesh.positions[i][0];
//     meshVertexArray[idx+1] = mesh.positions[i][1];
//     meshVertexArray[idx+2] = mesh.positions[i][2];
//     meshVertexArray[idx+3] = 1;
//     meshVertexArray[idx+4] = 0;
//     meshVertexArray[idx+5] = 0;
//     idx += 6;
// }
