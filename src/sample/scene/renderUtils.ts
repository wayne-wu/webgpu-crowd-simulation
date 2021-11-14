// Create a vertex buffer from the given data
export const getVerticesBuffer = (device: GPUDevice, vertexArray: Float32Array) => {
    const vertexBuffer = device.createBuffer({
        size: vertexArray.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
  new Float32Array(vertexBuffer.getMappedRange()).set(vertexArray);
  vertexBuffer.unmap();
  return vertexBuffer;
}

// Create a pipeline given the parameters
export const getPipeline = (device: GPUDevice, code, vertEntryPoint: string, fragEntryPoint: string, 
                         arrayStride: number, posOffset: number, uvOffset: number, presentationFormat, primitiveType, cullMode) => {
  const pipeline = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: code,
      }),
      entryPoint: vertEntryPoint,
      buffers: [
        {
          arrayStride: arrayStride,
          attributes: [
            {
              // position
              shaderLocation: 0,
              offset: posOffset,
              format: 'float32x4',
            },
            {
              // uv
              shaderLocation: 1,
              offset: uvOffset,
              format: 'float32x2',
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: code,
      }),
      entryPoint: fragEntryPoint,
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: primitiveType,
      cullMode: cullMode,
    },

    // Enable depth testing so that the fragment closest to the camera
    // is rendered in front.
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });
  return pipeline;
}


export const getDepthTexture = (device: GPUDevice, presentationSize) => {
  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return depthTexture;
}

export const getUniformBuffer = (device: GPUDevice) => {
  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return uniformBuffer;
};
  
export const getUniformBindGroup = (device: GPUDevice, pipeline: GPURenderPipeline, uniformBuffer: GPUBuffer) => {
  const uniformBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });
  return uniformBindGroup;
}

  

//   const renderPassDescriptor: GPURenderPassDescriptor = {
//     colorAttachments: [
//       {
//         view: undefined, // Assigned later

//         loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
//         storeOp: 'store',
//       },
//     ],
//     depthStencilAttachment: {
//       view: depthTexture.createView(),

//       depthLoadValue: 1.0,
//       depthStoreOp: 'store',
//       stencilLoadValue: 0,
//       stencilStoreOp: 'store',
//     },
//   };