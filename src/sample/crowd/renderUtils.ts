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

// TODO: There's probably a way to combine getCrowdRenderPipeline() with getPipeline()
export const getCrowdRenderPipeline = (device: GPUDevice, code, arrayStride: number, posOffset: number, colOffset: number, 
  vertArrayStride: number, vertPosOffset: number, vertUVOffset: number, presentationFormat) => {
  const renderPipelineCrowd = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: code,
      }),
      entryPoint: 'vs_main',
      buffers: [
        {
          // instanced agents buffer
          arrayStride: arrayStride,
          stepMode: 'instance',
          attributes: [
            {
              // position
              shaderLocation: 0,
              offset: posOffset,
              format: 'float32x3',
            },
            {
              // color
              shaderLocation: 1,
              offset: colOffset,
              format: 'float32x4',
            },
          ],
        },
        {
          arrayStride: vertArrayStride,
          attributes: [
            {
              // position
              shaderLocation: 2,
              offset: vertPosOffset,
              format: 'float32x4',
            },
            {
              // uv
              shaderLocation: 3,
              offset: vertUVOffset,
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
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },

    depthStencil: {
      depthWriteEnabled: false,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });
  return renderPipelineCrowd;
};

export const getDepthTexture = (device: GPUDevice, presentationSize) => {
  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return depthTexture;
}

export const getUniformBuffer = (device: GPUDevice, size: number) => {
  const uniformBufferSize = size; // 4x4 matrix
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