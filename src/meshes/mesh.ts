import {GLTFLoader} from '../../node_modules/three/examples/jsm/loaders/GLTFLoader.js';

export class Mesh {
    vertexArray : Float32Array;
    vertexCount : number;
    itemSize : number;
    posOffset : number;
    uvOffset : number;

    constructor(array : Array<number>, count : number){
        this.vertexArray = new Float32Array(array);
        this.vertexCount = count;
        this.itemSize = 4 * 4 + // position vec4
                        2 * 4;  // uv vec2
        this.posOffset = 0;
        this.uvOffset = 4 * 4;
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
                    let gltfIdxArray = child.geometry.index.array;
                    let vertArrayIdx = 0;

                    console.log(gltfIdxArray);

                    for (let i = 0; i < gltfIdxArray.length; i++){
                        let idx = gltfIdxArray[i];
                        tmpMeshVertexArray[vertArrayIdx+0] = gltfArray[idx * 3 + 0];
                        tmpMeshVertexArray[vertArrayIdx+1] = gltfArray[idx * 3 + 1];
                        tmpMeshVertexArray[vertArrayIdx+2] = gltfArray[idx * 3 + 2];
                        tmpMeshVertexArray[vertArrayIdx+3] = 1;
                        tmpMeshVertexArray[vertArrayIdx+4] = 1; // tmp UV coords
                        tmpMeshVertexArray[vertArrayIdx+5] = 1;
                        vertArrayIdx += 6;
                    }

                    let mesh = new Mesh(tmpMeshVertexArray, vertCount);
                    resolve(mesh);
                }
            })
        });
    })
    return promise;
}