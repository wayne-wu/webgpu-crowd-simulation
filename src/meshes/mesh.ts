import { mat4 } from 'gl-matrix';
import {GLTFLoader} from '../../node_modules/three/examples/jsm/loaders/GLTFLoader.js';

export class Mesh {
    vertexArray : Float32Array;
    vertexCount : number;
    itemSize : number;
    posOffset : number;
    uvOffset : number;
    normalOffset : number;
    scale: number;
    rotation: mat4;

    constructor(array : Array<number>, count : number){
        this.vertexArray = new Float32Array(array);
        this.vertexCount = count;
        this.itemSize = 4 * 4 + // position vec4
                        2 * 4 + // uv vec2
                        4 * 4;  // normal vec4
        this.posOffset = 0;
        this.uvOffset = 4 * 4;
        this.normalOffset = 6 * 4;
        this.scale = 0;                 // set in main, specific to model
        this.rotation = mat4.create();  // set in main, specific to model
    }
}

export function loadModel(gltfPath : string) {
    let tmpMeshVertexArray = [];

    let promise = new Promise((resolve, reject) => {
        let loader = new GLTFLoader();

        loader.load( gltfPath, ( gltf ) => {
            // extract vertices and save to Float32Array
            gltf.scene.traverse( ( child ) => {
                if ( child.isMesh ) {
                
                    //let vertCount = child.geometry.attributes.position.count;
                    let vertCount = child.geometry.index.count;
                    let gltfArray = child.geometry.attributes.position.array;
                    let gltfUVArray = child.geometry.attributes.uv.array;
                    let gltfNormalArray = child.geometry.attributes.normal.array;
                    let gltfIdxArray = child.geometry.index.array;
                    let vertArrayIdx = 0;

                    for (let i = 0; i < gltfIdxArray.length; i++){
                        let idx = gltfIdxArray[i];
                        tmpMeshVertexArray[vertArrayIdx+0] = gltfArray[idx * 3 + 0];        // position.x
                        tmpMeshVertexArray[vertArrayIdx+1] = gltfArray[idx * 3 + 1];        // position.y
                        tmpMeshVertexArray[vertArrayIdx+2] = gltfArray[idx * 3 + 2];        // position.z
                        tmpMeshVertexArray[vertArrayIdx+3] = 1;                             // position.w
                        tmpMeshVertexArray[vertArrayIdx+4] = 1;                             // uv.u
                        tmpMeshVertexArray[vertArrayIdx+5] = 1;                             // uv.v
                        tmpMeshVertexArray[vertArrayIdx+6] = gltfNormalArray[idx * 3 + 0];  // normal.x
                        tmpMeshVertexArray[vertArrayIdx+7] = gltfNormalArray[idx * 3 + 1];  // normal.y
                        tmpMeshVertexArray[vertArrayIdx+8] = gltfNormalArray[idx * 3 + 2];  // normal.z
                        tmpMeshVertexArray[vertArrayIdx+9] = 0;                             // normal.w

                        vertArrayIdx += 10;
                    }

                    let mesh = new Mesh(tmpMeshVertexArray, vertCount);
                    resolve(mesh);
                }
            })
        });
    })
    return promise;
}