
import {
  platformVertexArray,
  platformVertexSize,
  platformUVOffset,
  platformPositionOffset,
  platformVertexCount,
} from '../../meshes/platform';

import {
  getGridLines,
  gridLinesVertexSize,
  gridLinesVertexCount,
  gridLinesPositionOffset,
  gridLinesUVOffset
} from '../../meshes/gridLines';

import {
  cubeVertexArray,
  cubeVertexSize,
  cubeUVOffset,
  cubePositionOffset,
  cubeVertexCount,
} from '../../meshes/cube';

import renderWGSL from './shaders.wgsl';
import crowdWGSL from './crowd.wgsl';
import obstaclesWGSL from '../../shaders/obstacles.render.wgsl'
import { mat4 } from 'gl-matrix';
import { ComputeBufferManager } from './crowdUtils';

export class RenderBufferManager {

  device : GPUDevice;

  platformVertexBuffer    : GPUBuffer;
  gridLinesVertexBuffer   : GPUBuffer;
  prototypeVertexBuffer   : GPUBuffer;
  obstacleVertexBuffer    : GPUBuffer;

  platformPipeline        : GPURenderPipeline;
  gridLinesPipeline       : GPURenderPipeline;
  crowdPipeline           : GPURenderPipeline;
  obstaclesPipeline       : GPURenderPipeline;

  commonUniformBuffer     : GPUBuffer;
  platformUniformBuffer   : GPUBuffer;
  gridLinesUniformBuffer  : GPUBuffer;
  crowdUniformBuffer      : GPUBuffer;
  obstaclesUniformBuffer  : GPUBuffer;

  platformBindGroup       : GPUBindGroup;
  gridLinesBindGroup      : GPUBindGroup;
  crowdBindGroup          : GPUBindGroup;
  obstaclesBindGroup      : GPUBindGroup;

  renderPassDescriptor    : GPURenderPassDescriptor;

  constructor (device: GPUDevice, gridWidth: number, presentationFormat: GPUTextureFormat, presentationSize, cbm: ComputeBufferManager) 
  {
    this.device = device;
    this.initBuffers(gridWidth);
    this.buildPipelines(presentationFormat, cbm);
    this.setBindGroups();
    this.setRenderPassDescriptor(presentationSize);
  }

  initBuffers(gridWidth: number) {
    // Create vertex buffers for the platform and the grid lines
    this.platformVertexBuffer = getVerticesBuffer(this.device, platformVertexArray);

    // Compute the grid lines based on an input gridWidth
    let gridLinesVertexArray = getGridLines(gridWidth);
    this.gridLinesVertexBuffer = getVerticesBuffer(this.device, gridLinesVertexArray);

    this.prototypeVertexBuffer = getVerticesBuffer(this.device, cubeVertexArray);
    this.obstacleVertexBuffer = getVerticesBuffer(this.device, cubeVertexArray);
  }

  buildPipelines(presentationFormat, cbm: ComputeBufferManager) {

    this.platformPipeline = getPipeline(
      this.device, renderWGSL, 'vs_main', 'fs_platform', platformVertexSize,
      platformPositionOffset, platformUVOffset, presentationFormat, 'triangle-list', 'back'
    );

    this.gridLinesPipeline = getPipeline(
      this.device, renderWGSL, 'vs_main', 'fs_gridLines', gridLinesVertexSize,
      gridLinesPositionOffset, gridLinesUVOffset, presentationFormat, 'line-list', 'none'
    );

    this.crowdPipeline = getCrowdRenderPipeline(
      this.device, crowdWGSL, cbm.agentInstanceSize, cbm.agentPositionOffset, 
      cbm.agentColorOffset, cubeVertexSize, cubePositionOffset, cubeUVOffset, presentationFormat
    );

    this.obstaclesPipeline = getObstaclesRenderPipeline(
      this.device, obstaclesWGSL, cbm.obstacleInstanceSize, 0, 
      cubeVertexSize, cubePositionOffset, cubeUVOffset, presentationFormat);
  }

  setBindGroups() {
    // NOTE: Is there really no way to share the same uniform buffer across different pipelines?
    // Seems very efficient to have to redeclare pretty much the same data multiple times
    let mvpSize = 4 * 16;  // mat4
    this.platformUniformBuffer = getUniformBuffer(this.device, mvpSize);
    this.gridLinesUniformBuffer = getUniformBuffer(this.device, mvpSize);
    this.crowdUniformBuffer = getUniformBuffer(this.device, mvpSize);
    this.obstaclesUniformBuffer = getUniformBuffer(this.device, mvpSize);

    this.platformBindGroup = getUniformBindGroup(this.device, this.platformPipeline, this.platformUniformBuffer);
    this.gridLinesBindGroup = getUniformBindGroup(this.device, this.gridLinesPipeline, this.gridLinesUniformBuffer);
    this.crowdBindGroup = getUniformBindGroup(this.device, this.crowdPipeline, this.crowdUniformBuffer);
    this.obstaclesBindGroup = getUniformBindGroup(this.device, this.obstaclesPipeline, this.obstaclesUniformBuffer);
  }
  
  setRenderPassDescriptor(presentationSize) {
    // Get the depth texture for both pipelines
    const depthTexture = getDepthTexture(this.device, presentationSize);

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned later

          loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),

        depthLoadValue: 1.0,
        depthStoreOp: 'store',
        stencilLoadValue: 0,
        stencilStoreOp: 'store',
      },
    };
  }

  resetGridLinesBuffer(gridWidth: number) {
    // Compute the grid lines based on an input gridWidth
    let gridLinesVertexArray = getGridLines(gridWidth);
    this.gridLinesVertexBuffer = getVerticesBuffer(this.device, gridLinesVertexArray);
  }

  drawPlatform(device: GPUDevice, transformationMatrix: Float32Array, passEncoder: GPURenderPassEncoder) {
    device.queue.writeBuffer(
      this.platformUniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    passEncoder.setPipeline(this.platformPipeline);
    passEncoder.setBindGroup(0, this.platformBindGroup);
    passEncoder.setVertexBuffer(0, this.platformVertexBuffer);
    passEncoder.draw(platformVertexCount, 1, 0, 0);
  }

  drawGridLines(device: GPUDevice, transformationMatrix: Float32Array, passEncoder: GPURenderPassEncoder) {
    device.queue.writeBuffer(
      this.gridLinesUniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    passEncoder.setPipeline(this.gridLinesPipeline);
    passEncoder.setBindGroup(0, this.gridLinesBindGroup);
    passEncoder.setVertexBuffer(0, this.gridLinesVertexBuffer);
    passEncoder.draw(gridLinesVertexCount, 1, 0, 0);
  }

  drawCrowd(device: GPUDevice, mvp: Float32Array, passEncoder: GPURenderPassEncoder, agentsBuffer: GPUBuffer, numAgents: number) {
    device.queue.writeBuffer(
      this.crowdUniformBuffer,
      0,
      mvp.buffer,
      mvp.byteOffset,
      mvp.byteLength
    );
    passEncoder.setPipeline(this.crowdPipeline);
    passEncoder.setBindGroup(0, this.crowdBindGroup);
    passEncoder.setVertexBuffer(0, agentsBuffer);
    passEncoder.setVertexBuffer(1, this.prototypeVertexBuffer);
    passEncoder.draw(cubeVertexCount, numAgents, 0, 0);
  }
  
  drawObstacles(device: GPUDevice, mvp: Float32Array, passEncoder: GPURenderPassEncoder, obstaclesBuffer: GPUBuffer, numObstacles: number) {
    device.queue.writeBuffer(
      this.obstaclesUniformBuffer,
      0,
      mvp.buffer,
      mvp.byteOffset,
      mvp.byteLength
    );
    passEncoder.setPipeline(this.obstaclesPipeline);
    passEncoder.setBindGroup(0, this.obstaclesBindGroup);
    passEncoder.setVertexBuffer(0, obstaclesBuffer);
    passEncoder.setVertexBuffer(1, this.obstacleVertexBuffer);
    passEncoder.draw(cubeVertexCount, numObstacles, 0, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////////////
//                         renderBufferManager Helpers                                //
////////////////////////////////////////////////////////////////////////////////////////

// Create a vertex buffer from the given data
const getVerticesBuffer = (device: GPUDevice, vertexArray: Float32Array) => {
  const vertexBuffer = device.createBuffer({
      size: vertexArray.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(vertexArray);
  vertexBuffer.unmap();
  return vertexBuffer;
}

const getPipelineDescriptor = (device: GPUDevice, code, vs: string, fs: string, 
  vertexBuffers, presentationFormat: GPUTextureFormat, primitiveType: GPUPrimitiveTopology, cullMode: GPUCullMode) => {

  let descriptor : GPURenderPipelineDescriptor = {
    vertex: {
      module: device.createShaderModule({code: code}),
      entryPoint: vs,
      buffers: vertexBuffers,
    },
    fragment : {
      module: device.createShaderModule({code: code}),
      entryPoint: fs,
      targets: [
        {
          format: presentationFormat,
        },
      ]
    },
    primitive: {
      topology: primitiveType,
      cullMode: cullMode,
    },
    // Enable depth testing so that the fragment closest to the camera
    // is rendered in front.
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: <GPUCompareFunction>'less',
      format: <GPUTextureFormat>'depth24plus',
    },
  };
  return descriptor;

}

// Create a pipeline given the parameters
const getPipeline = (device: GPUDevice, code, vertEntryPoint: string, fragEntryPoint: string, 
                            arrayStride: number, posOffset: number, uvOffset: number, presentationFormat, primitiveType, cullMode) => {
  let buffers = [
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
  ]
  
  const pipeline = device.createRenderPipeline(
    getPipelineDescriptor(device, code, vertEntryPoint, fragEntryPoint, buffers, presentationFormat, primitiveType, cullMode));
  return pipeline;
}

const getCrowdRenderPipeline = (device: GPUDevice, code, arrayStride: number, posOffset: number, colOffset: number, 
                                vertArrayStride: number, vertPosOffset: number, vertUVOffset: number, presentationFormat) => {

  let buffers = [
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
  ]

  const renderPipelineCrowd = device.createRenderPipeline(
    getPipelineDescriptor(device, code, "vs_main", "fs_main", buffers, presentationFormat, "triangle-list", "back"));

  return renderPipelineCrowd;
};

const getObstaclesRenderPipeline = (device: GPUDevice, code, arrayStride: number, posOffset: number, 
  vertArrayStride: number, vertPosOffset: number, vertUVOffset: number, presentationFormat) => {
  let buffers = [
    {
      arrayStride: arrayStride,
      stepMode: 'instance',
      attributes: [
        {
          // position
          shaderLocation: 0,
          offset: 0,
          format: 'float32x3',
        },
        {
          // rotation-y
          shaderLocation: 1,
          offset: 3*4,
          format: 'float32',
        },
        {
          // scale
          shaderLocation: 2,
          offset: 4*4,
          format: 'float32x3',
        },
      ],
    },
    {
      arrayStride: vertArrayStride,
      attributes: [
        {
          // position
          shaderLocation: 3,
          offset: vertPosOffset,
          format: 'float32x4',
        },
        {
          // uv
          shaderLocation: 4,
          offset: vertUVOffset,
          format: 'float32x2',
        },
      ],
    },
  ];

  const pipeline = device.createRenderPipeline(
    getPipelineDescriptor(device, code, "vs_main", "fs_main", buffers, presentationFormat, "triangle-list", "back"));
  return pipeline;
}

const getDepthTexture = (device: GPUDevice, presentationSize) => {
  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return depthTexture;
}

const getUniformBuffer = (device: GPUDevice, size: number) => {
  const uniformBufferSize = size; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
  size: uniformBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return uniformBuffer;
};

const getUniformBindGroup = (device: GPUDevice, pipeline: GPURenderPipeline, uniformBuffer: GPUBuffer) => {
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