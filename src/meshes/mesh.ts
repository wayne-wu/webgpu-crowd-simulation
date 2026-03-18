import { mat4 } from 'gl-matrix';
import {GLTFLoader} from '../../node_modules/three/examples/jsm/loaders/GLTFLoader.js';

export class Mesh {
    vertexArray     : Float32Array;
    indexArray      : Uint32Array;
    vertexCount     : number;
    indexCount      : number;
    itemSize        : number;
    posOffset       : number;
    uvOffset        : number;
    normalOffset    : number;
    colorOffset     : number;
    scale           : number;
    rotation        : mat4;

    constructor(vertexArray : Array<number> | Float32Array, indexArray? : Array<number> | Uint32Array){
        this.vertexArray = vertexArray instanceof Float32Array ? vertexArray : new Float32Array(vertexArray);
        this.vertexCount = this.vertexArray.length / 13;
        const defaultIndices = Array.from({ length: this.vertexCount }, (_, i) => i);
        const indices = indexArray == null ? defaultIndices : indexArray;
        this.indexArray = indices instanceof Uint32Array ? indices : new Uint32Array(indices);
        this.indexCount = this.indexArray.length;
        this.itemSize = 4 * 4 + // position vec4
                        2 * 4 + // uv vec2
                        4 * 4 + // normal vec4
                        3 * 4;  // color vec3
        this.posOffset = 0;
        this.uvOffset = 4 * 4;
        this.normalOffset = 6 * 4;
        this.colorOffset = 10 * 4;
        this.scale = 0;                 // set in main, specific to model
        this.rotation = mat4.create();  // set in main, specific to model
    }
}

export function loadModel(gltfPath : string, device: GPUDevice) {
    const tmpMeshVertexArray = [];
    const tmpMeshIndexArray = [];

    const promise = new Promise((resolve, reject) => {
        const loader = new GLTFLoader();

        loader.load( gltfPath, ( gltf ) => {
            let vertArrayIdx = 0;
            let vertexOffset = 0;

            // extract vertices and save to Float32Array
            gltf.scene.traverse( ( child ) => {
                if ( child.isMesh ) {
                    const gltfArray = child.geometry.attributes.position.array;
                    const gltfUVArray = child.geometry.attributes.uv.array;
                    const gltfNormalArray = child.geometry.attributes.normal.array;
                    const gltfIdxArray = child.geometry.index.array;
                    const color = child.material.color;

                    const vertexCount = child.geometry.attributes.position.count;
                    for (let i = 0; i < vertexCount; i++){
                        tmpMeshVertexArray[vertArrayIdx+0] = gltfArray[i * 3 + 0];         // position.x
                        tmpMeshVertexArray[vertArrayIdx+1] = gltfArray[i * 3 + 1];         // position.y
                        tmpMeshVertexArray[vertArrayIdx+2] = gltfArray[i * 3 + 2];         // position.z
                        tmpMeshVertexArray[vertArrayIdx+3] = 1;                             // position.w
                        tmpMeshVertexArray[vertArrayIdx+4] = gltfUVArray[i * 2 + 0];       // uv.u
                        tmpMeshVertexArray[vertArrayIdx+5] = gltfUVArray[i * 2 + 1];       // uv.v
                        tmpMeshVertexArray[vertArrayIdx+6] = gltfNormalArray[i * 3 + 0];   // normal.x
                        tmpMeshVertexArray[vertArrayIdx+7] = gltfNormalArray[i * 3 + 1];   // normal.y
                        tmpMeshVertexArray[vertArrayIdx+8] = gltfNormalArray[i * 3 + 2];   // normal.z
                        tmpMeshVertexArray[vertArrayIdx+9] = 0;                             // normal.w
                        tmpMeshVertexArray[vertArrayIdx+10] = color.r;                      // color.r
                        tmpMeshVertexArray[vertArrayIdx+11] = color.g;                      // color.g
                        tmpMeshVertexArray[vertArrayIdx+12] = color.b;                      // color.b
                        vertArrayIdx += 13;
                    }

                    for (let i = 0; i < gltfIdxArray.length; i++){
                        tmpMeshIndexArray.push(vertexOffset + gltfIdxArray[i]);
                    }

                    vertexOffset += vertexCount;
                }
            })
            const mesh = new Mesh(tmpMeshVertexArray, tmpMeshIndexArray);
            resolve(mesh);
        });
    })
    return promise;
}
