export const gridLinesVertexSize = 4 * 6; // Byte size of one gridLine vertex.
export const gridLinesPositionOffset = 0;
export const gridLinesUVOffset = 4 * 4;

export let gridLinesVertexCount = 0; // set in getGridLines function

// remap number from [0, gridWidth] to [-1, 1]
const remap = (n, gridWidth) => ((n / gridWidth) * 2 - 1);

export const getGridLines = gridWidth => {
    let vertexArray = [];
    let vertexCount = 0;
    let yVal = 1 + 0.1;
    
    // Draw horizontal lines
    for (let z = gridWidth; z >= 0; z--) {
        for (let x = 0; x < gridWidth; x++) {
            vertexArray[(vertexCount/2.0) * 12 + 0] = remap(x, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 1] = yVal;
            vertexArray[(vertexCount/2.0) * 12 + 2] = remap(z, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 3] = 1;
            vertexArray[(vertexCount/2.0) * 12 + 4] = 0;
            vertexArray[(vertexCount/2.0) * 12 + 5] = 0;

            vertexArray[(vertexCount/2.0) * 12 + 6] = remap(x + 1, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 7] = yVal;
            vertexArray[(vertexCount/2.0) * 12 + 8] = remap(z, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 9] = 1;
            vertexArray[(vertexCount/2.0) * 12 + 10] = 0;
            vertexArray[(vertexCount/2.0) * 12 + 11] = 0;
            vertexCount += 2;
        }
    }

    // Draw vertical lines
    for (let x = 0; x <= gridWidth; x++) {
        for (let z = gridWidth; z > 0; z--) {       
            vertexArray[(vertexCount/2.0) * 12 + 0] = remap(x, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 1] = yVal;
            vertexArray[(vertexCount/2.0) * 12 + 2] = remap(z, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 3] = 1;
            vertexArray[(vertexCount/2.0) * 12 + 4] = 0;
            vertexArray[(vertexCount/2.0) * 12 + 5] = 0;

            vertexArray[(vertexCount/2.0) * 12 + 6] = remap(x, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 7] = yVal;
            vertexArray[(vertexCount/2.0) * 12 + 8] = remap(z - 1, gridWidth);
            vertexArray[(vertexCount/2.0) * 12 + 9] = 1;
            vertexArray[(vertexCount/2.0) * 12 + 10] = 0;
            vertexArray[(vertexCount/2.0) * 12 + 11] = 0;
            vertexCount += 2;
        }
    }

    const gridLinesVertexArray = new Float32Array(vertexArray);
    gridLinesVertexCount = vertexCount;

    return gridLinesVertexArray;
};