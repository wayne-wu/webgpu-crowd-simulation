
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
import { mat4 } from 'gl-matrix';
import { Mesh } from '../../meshes/mesh';

import { vec4 } from 'gl-matrix';

export class renderBufferManager {

  device : GPUDevice;

  platformVertexBuffer    : GPUBuffer;
  gridLinesVertexBuffer   : GPUBuffer;
  prototypeVertexBuffer   : GPUBuffer;
  meshVertexBuffer        : GPUBuffer;

  platformPipeline        : GPURenderPipeline;
  gridLinesPipeline       : GPURenderPipeline;
  crowdPipeline           : GPURenderPipeline;

  platformUniformBuffer   : GPUBuffer;
  gridLinesUniformBuffer  : GPUBuffer;
  crowdUniformBuffer      : GPUBuffer;
  u_CameraPosBuffer       : GPUBuffer;
  u_AgentScale            : GPUBuffer;

  platformBindGroup       : GPUBindGroup;
  gridLinesBindGroup      : GPUBindGroup;
  crowdBindGroup          : GPUBindGroup;

  bindGroupLayout         : GPUBindGroupLayout;

  renderPassDescriptor    : GPURenderPassDescriptor;

  mesh : Mesh;

  constructor (device: GPUDevice, gridWidth: number, presentationFormat, presentationSize,
               agentInstanceByteSize: number, agentPositionOffset: number, agentColorOffset: number, 
               agentVelocityOffset: number, mesh : Mesh, cameraPos: vec4) 
  {
    this.device = device;
    this.mesh = mesh;
    this.initBuffers(gridWidth);
    
    this.setBindGroupLayout();
    this.buildPipelines(presentationFormat, agentInstanceByteSize, agentPositionOffset, agentColorOffset, agentVelocityOffset);  
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

    this.meshVertexBuffer = getVerticesBuffer(this.device, this.mesh.vertexArray);
  }

  buildPipelines(presentationFormat, agentInstanceByteSize: number, agentPositionOffset: number, 
                agentColorOffset: number, agentVelocityOffset: number) {

    this.platformPipeline = getPipeline(
      this.device, renderWGSL, 'vs_main', 'fs_platform', platformVertexSize,
      platformPositionOffset, platformUVOffset, presentationFormat, 'triangle-list', 'back'
    );

    this.gridLinesPipeline = getPipeline(
      this.device, renderWGSL, 'vs_main', 'fs_gridLines', gridLinesVertexSize,
      gridLinesPositionOffset, gridLinesUVOffset, presentationFormat, 'line-list', 'none'
    );

    this.crowdPipeline = getCrowdRenderPipeline(
      this.device, crowdWGSL, agentInstanceByteSize, agentPositionOffset, 
      agentColorOffset, agentVelocityOffset, this.mesh.normalOffset, this.mesh.itemSize, this.mesh.posOffset, 
      this.mesh.uvOffset, presentationFormat, this.bindGroupLayout
    );
  }

  setBindGroups() {
    this.platformUniformBuffer = getUniformBuffer(this.device, 4 * 16);
    this.gridLinesUniformBuffer = getUniformBuffer(this.device, 4 * 16);
    this.crowdUniformBuffer = getUniformBuffer(this.device, 4 * 16);
    this.u_CameraPosBuffer = getUniformBuffer(this.device, 4 * 4);
    this.u_AgentScale = getUniformBuffer(this.device, 4 * 1);

    this.platformBindGroup = getUniformBindGroup(this.device, this.platformPipeline, this.platformUniformBuffer);
    this.gridLinesBindGroup = getUniformBindGroup(this.device, this.gridLinesPipeline, this.gridLinesUniformBuffer);
    this.crowdBindGroup = getCrowdUniformBindGroup(this.device, this.crowdPipeline, this.crowdUniformBuffer, this.u_CameraPosBuffer, this.u_AgentScale);
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

  drawGridLines(device: GPUDevice, transformationMatrix: Float32Array, passEncoder: GPURenderPassEncoder, gridOn: boolean) {
    if (gridOn){
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
  }

  drawCrowd(device: GPUDevice, mvp: mat4, passEncoder: GPURenderPassEncoder, agentsBuffer: GPUBuffer, numAgents: number, cameraPos: vec4) {
      device.queue.writeBuffer(
        this.crowdUniformBuffer,
        0,
          new Float32Array([
          // modelViewProjectionMatrix
          mvp[0],  mvp[1],  mvp[2],  mvp[3],
          mvp[4],  mvp[5],  mvp[6],  mvp[7],
          mvp[8],  mvp[9],  mvp[10], mvp[11],
          mvp[12], mvp[13], mvp[14], mvp[15],
        ])
      );
      device.queue.writeBuffer(
        this.u_CameraPosBuffer,
        0,
          new Float32Array([
          // camera position
          cameraPos[0], cameraPos[1], cameraPos[2], cameraPos[3]
        ])
      );
      device.queue.writeBuffer(
        this.u_AgentScale,
        0,
          new Float32Array([
          // agent scale
          this.mesh.scale
        ])
      );
      passEncoder.setPipeline(this.crowdPipeline);
      passEncoder.setBindGroup(0, this.crowdBindGroup);
      passEncoder.setVertexBuffer(0, agentsBuffer);
      passEncoder.setVertexBuffer(1, this.meshVertexBuffer);
      passEncoder.draw(this.mesh.vertexCount, numAgents, 0, 0);
      passEncoder.endPass();
  }  

  setBindGroupLayout = () => {
    // create bindgroup layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0, // viewproj matrix
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "uniform"
          }
        },
        {
          binding: 1, // camera position
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform"
          }
        },
        {
          binding: 2, // agent scale
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "uniform"
          }
        },
      ]
    });
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

// Create a pipeline given the parameters
const getPipeline = (device: GPUDevice, code, vertEntryPoint: string, fragEntryPoint: string, 
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
const getCrowdRenderPipeline = (device: GPUDevice, code, arrayStride: number, posOffset: number, colOffset: number, velOffset: number, vertNorOffset: number,
                                       vertArrayStride: number, vertPosOffset: number, vertUVOffset: number, presentationFormat, bindGroupLayout) => {
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
        {
          // orientation
          shaderLocation: 2,
          offset: velOffset,
          format: 'float32x4'
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
        {
          // normal
          shaderLocation: 5,
          offset: vertNorOffset,
          format: 'float32x4'
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
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout]}),
    primitive: {
      topology: 'triangle-list',
      frontFace: 'cw'
    },

    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });
  return renderPipelineCrowd;
};

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

const getUniformBindGroup = (device: GPUDevice, pipeline: GPURenderPipeline, uniformBuffer1: GPUBuffer) => {
  const uniformBindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer1,
      },
    },
  ],
  });
  return uniformBindGroup;
}

const getCrowdUniformBindGroup = (device: GPUDevice, pipeline: GPURenderPipeline, uniformBuffer1: GPUBuffer, uniformBuffer2: GPUBuffer, uniformBuffer3: GPUBuffer) => {
  const uniformBindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer1,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: uniformBuffer2,
      },
    },
    {
      binding: 2,
      resource: {
        buffer: uniformBuffer3
      }
    }
  ],
  });
  return uniformBindGroup;
}