
import {
  platformVertexArray,
  platformVertexSize,
  platformUVOffset,
  platformPositionOffset,
  platformVertexCount,
  platformNorOffset,
} from '../../meshes/platform';

import {
  cubeVertexArray,
  cubeVertexSize,
  cubeUVOffset,
  cubePositionOffset,
  cubeVertexCount,
  cubeNorOffset
} from '../../meshes/cube';

import { 
  sphereItemSize, 
  sphereNorOffset, 
  spherePosOffset, 
  sphereVertCount, 
  sphereVertexArray 
} from '../../meshes/sphere';


import renderWGSL from '../../shaders/background.render.wgsl';
import crowdWGSL from '../../shaders/crowd.render.wgsl';
import crowdShadowWGSL from '../../shaders/crowdShadow.render.wgsl';
import obstaclesWGSL from '../../shaders/obstacles.render.wgsl';
import headerWGSL from '../../shaders/header.render.wgsl';
import { ComputeBufferManager } from './crowdUtils';
import { Mesh } from '../../meshes/mesh';

import { vec3, mat4 } from 'gl-matrix';
import Camera from "./Camera";

const shadowDepthTextureSize = 2048;

export class RenderBufferManager {

  device : GPUDevice;

  platformVertexBuffer    : GPUBuffer;
  obstacleVertexBuffer    : GPUBuffer;
  meshVertexBuffer        : GPUBuffer;
  goalVertBuffer          : GPUBuffer;

  platformPipeline        : GPURenderPipeline;
  crowdPipeline           : GPURenderPipeline;
  crowdShadowPipeline     : GPURenderPipeline;
  obstaclesPipeline       : GPURenderPipeline;
  goalPipeline            : GPURenderPipeline;

  sceneUBO                : GPUBuffer;
  platformModelUBO        : GPUBuffer;
  crowdModelUBO           : GPUBuffer;

  platformModelBindGroup  : GPUBindGroup;
  crowdModelBindGroup     : GPUBindGroup;
  crowdShadowBindGroup    : GPUBindGroup;
  platformBindGroup       : GPUBindGroup;
  crowdBindGroup          : GPUBindGroup;
  obstaclesBindGroup      : GPUBindGroup;
  goalBindGroup           : GPUBindGroup;

  uboBindGroupLayout      : GPUBindGroupLayout;
  platformBindGroupLayout : GPUBindGroupLayout;
  shadowBindGroupLayout   : GPUBindGroupLayout;

  renderPassDescriptor    : GPURenderPassDescriptor;
  shadowPassDescriptor    : GPURenderPassDescriptor;

  shadowDepthTexture      : GPUTexture;
  shadowDepthTextureView  : GPUTextureView;

  mesh                    : Mesh;

  constructor (device: GPUDevice, gridWidth: number, presentationFormat: GPUTextureFormat, presentationSize, cbm: ComputeBufferManager, 
               mesh : Mesh, gridTexture: GPUTexture, sampler, showGoals: boolean) 
  {
    this.device = device;
    this.mesh = mesh;
    this.initVBOs(gridWidth, showGoals);
    this.initBindGroupLayouts();
    this.initUBOs();
    this.initRenderPasses(presentationSize);
    this.initBindGroups(gridTexture, sampler);
    this.initPipelines(presentationFormat, cbm);
  }

  initVBOs(gridWidth: number, showGoals: boolean) {
    this.platformVertexBuffer = createVBO(this.device, platformVertexArray);
    this.obstacleVertexBuffer = createVBO(this.device, cubeVertexArray);
    this.meshVertexBuffer = createVBO(this.device, this.mesh.vertexArray);
    this.goalVertBuffer = createVBO(this.device, new Float32Array(sphereVertexArray));
  }

  initBindGroupLayouts() {
    // Simple bind group layout
    this.uboBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'uniform',
          },
        },
      ],
    });

    // Bind group layout with texture and sampler
    this.platformBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          sampler: {
            type: 'filtering',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'float',
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          sampler: {
            type: 'comparison',
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'depth',
          },
        },
      ],
    });

    // Bind group layout with texture and sampler
    this.shadowBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          sampler: {
            type: 'comparison',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'depth',
          },
        },
      ],
    });
  }

  initUBOs() {
    // Scene UBO
    const sceneBufferSize = 4 * 16 + 4 * 16 + 3 * 4 + 1 * 4 + 3 * 4 + 1 * 4;
    this.sceneUBO = createUBO(this.device, sceneBufferSize);
    this.platformModelUBO = createUBO(this.device, 4 * 16);
    this.crowdModelUBO = createUBO(this.device, 4 * 16);

    // Initialize all the non-changing UBO data
    {
      const upVector = vec3.fromValues(0, 1, 0);
      const origin = vec3.fromValues(0, 0, 0);  
      const lightPosition = vec3.fromValues(50, 50, -50);
      const lightViewMatrix = mat4.create();
      mat4.lookAt(lightViewMatrix, lightPosition, origin, upVector);
    
      const lightProjectionMatrix = mat4.create();
      {
        // TODO: Need to find the right setting for this
        const margin = 20;
        const left = -50;
        const right = 20;
        const bottom = -30;
        const top = 30;
        const near = 1.0;
        const far = 150.0; 
        mat4.ortho(lightProjectionMatrix, left, right, bottom, top, near, far);
      }
    
      const lightViewProjMatrix = mat4.create();
      mat4.multiply(lightViewProjMatrix, lightProjectionMatrix, lightViewMatrix);
      this.device.queue.writeBuffer(this.sceneUBO, 0, lightViewProjMatrix as Float32Array);
      this.device.queue.writeBuffer(this.sceneUBO, 2 * 4 * 16, lightPosition as Float32Array);
    }
    {
      const modelMatrix = mat4.create();
      mat4.identity(modelMatrix);
      mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(this.mesh.scale, this.mesh.scale, this.mesh.scale));
      // mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0.0, -1.0, 0.0));
      this.device.queue.writeBuffer(this.crowdModelUBO, 0, modelMatrix as Float32Array);
    }
  }

  initPipelines(presentationFormat, cbm: ComputeBufferManager) {
    this.initPlatformPipeline(presentationFormat);
    this.initCrowdShadowPipeline(presentationFormat, cbm);
    this.initCrowdPipeline(presentationFormat, cbm);
    this.initObstaclePipeline(presentationFormat, cbm);
    this.initGoalPipeline(presentationFormat, cbm);
  }

  initBindGroups(gridTexture: GPUTexture, sampler: GPUSampler) {  
    //group = 0, binding = 0
    this.platformBindGroup = this.device.createBindGroup({
      layout: this.platformBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.sceneUBO,
          },
        },
        {
          binding: 1,
          resource: sampler,
        },
        {
          binding: 2,
          resource: gridTexture.createView(),
        },
        {
          binding: 3,
          resource: this.device.createSampler({compare: 'less',}),
        },
        {
          binding: 4,
          resource: this.shadowDepthTextureView,
        },
      ],
      });
    //group = 1, binding = 0
    this.platformModelBindGroup = createBindGroup(this.device, this.uboBindGroupLayout, this.platformModelUBO);
    
    this.crowdShadowBindGroup = createBindGroup(this.device, this.uboBindGroupLayout, this.sceneUBO);
    this.crowdBindGroup = createTexturedBindGroup(this.device, this.shadowBindGroupLayout, this.sceneUBO, 
      this.shadowDepthTextureView, this.device.createSampler({compare: 'less',}))
    this.crowdModelBindGroup = createBindGroup(this.device, this.uboBindGroupLayout, this.crowdModelUBO);

    this.obstaclesBindGroup = createBindGroup(this.device, this.uboBindGroupLayout, this.sceneUBO);
    this.goalBindGroup = createBindGroup(this.device, this.uboBindGroupLayout, this.sceneUBO);
  }
  
  initRenderPasses(presentationSize) {
    // Get the depth texture for both pipelines
    const depthTexture = getDepthTexture(this.device, presentationSize);

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned later

          loadValue: { r: 0.89, g: 0.92, b: 1.0, a: 1.0 },
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

    this.shadowDepthTexture = this.device.createTexture({
      size: [shadowDepthTextureSize, shadowDepthTextureSize, 1],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      format: 'depth32float',
    });
    this.shadowDepthTextureView = this.shadowDepthTexture.createView();

    this.shadowPassDescriptor = {
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowDepthTextureView,
        depthLoadValue: 1.0,
        depthStoreOp: 'store',
        stencilLoadValue: 0,
        stencilStoreOp: 'store',
      },
    };
  }

  initPlatformPipeline(presentationFormat) {
    const buffers = [
      {
        arrayStride: platformVertexSize,
        attributes: [
          {
            // position
            shaderLocation: 0,
            offset: platformPositionOffset,
            format: 'float32x4',
          },
          {
            // uv
            shaderLocation: 1,
            offset: platformUVOffset,
            format: 'float32x2',
          },
          {
            // normal
            shaderLocation: 2,
            offset: 40,
            format: 'float32x4',
          }
        ],
      },
    ]
    const layout = this.device.createPipelineLayout({
      bindGroupLayouts : [this.platformBindGroupLayout, this.uboBindGroupLayout],
    });
    var desc = getPipelineDescriptor(this.device, layout, renderWGSL, 'vs_main', 'fs_platform', 
      buffers, presentationFormat, 'triangle-list', 'back');

    this.platformPipeline = this.device.createRenderPipeline(desc)
  }

  initCrowdPipeline(presentationFormat, cbm : ComputeBufferManager) {
    const buffers = [
      {
        // instanced agents buffer
        arrayStride: cbm.agentInstanceSize,
        stepMode: 'instance',
        attributes: [
          {
            // position
            shaderLocation: 0,
            offset: cbm.agentPositionOffset,
            format: 'float32x3',
          },
          {
            // color
            shaderLocation: 1,
            offset: cbm.agentColorOffset,
            format: 'float32x4',
          },
          {
            // velocity
            shaderLocation: 2,
            offset: cbm.agentVelocityOffset,
            format: 'float32x3'
          },
        ],
      },
      {
        arrayStride: this.mesh.itemSize,
        attributes: [
          {
            // position
            shaderLocation: 3,
            offset: this.mesh.posOffset,
            format: 'float32x4',
          },
          {
            // uv
            shaderLocation: 4,
            offset: this.mesh.uvOffset,
            format: 'float32x2',
          },
          {
            // normal
            shaderLocation: 5,
            offset: this.mesh.normalOffset,
            format: 'float32x4'
          },
          {
            // mesh color
            shaderLocation: 6,
            offset: this.mesh.colorOffset,
            format: 'float32x3'
          }
          ],
      },   
    ];

    const layout = this.device.createPipelineLayout({
      bindGroupLayouts : [this.shadowBindGroupLayout, this.uboBindGroupLayout],
    });
    var desc = getPipelineDescriptor(this.device, layout, crowdWGSL, 'vs_main', 'fs_main', 
      buffers, presentationFormat, 'triangle-list', 'back');
    desc.primitive.frontFace = 'cw';
    this.crowdPipeline = this.device.createRenderPipeline(desc);
  }

  initCrowdShadowPipeline(presentationFormat, cbm : ComputeBufferManager) {
    const buffers = [
      {
        // instanced agents buffer
        arrayStride: cbm.agentInstanceSize,
        stepMode: 'instance',
        attributes: [
          {
            // position
            shaderLocation: 0,
            offset: cbm.agentPositionOffset,
            format: 'float32x3',
          },
          {
            // velocity
            shaderLocation: 1,
            offset: cbm.agentVelocityOffset,
            format: 'float32x3'
          },
        ],
      },
      {
        arrayStride: this.mesh.itemSize,
        attributes: [
          {
            // position
            shaderLocation: 2,
            offset: this.mesh.posOffset,
            format: 'float32x4',
          },
          ],
      },   
    ];

    const layout = this.device.createPipelineLayout({
      bindGroupLayouts : [this.uboBindGroupLayout, this.uboBindGroupLayout],
    });
    var desc = getPipelineDescriptor(this.device, layout, crowdShadowWGSL, 'vs_main', 'fs_main', 
      buffers, presentationFormat, 'triangle-list', 'back');
    desc.primitive.frontFace = 'cw';
    desc.fragment.targets = [];
    desc.depthStencil.format = 'depth32float';
    this.crowdShadowPipeline = this.device.createRenderPipeline(desc);
  }

  initObstaclePipeline(presentationFormat, cbm : ComputeBufferManager) {
    const buffers = [
      {
        arrayStride: cbm.obstacleInstanceSize,
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
        arrayStride: cubeVertexSize,
        attributes: [
          {
            // position
            shaderLocation: 3,
            offset: cubePositionOffset,
            format: 'float32x4',
          },
          {
            // uv
            shaderLocation: 4,
            offset: cubeUVOffset,
            format: 'float32x2',
          },
          {
            // normal
            shaderLocation: 5,
            offset: cubeNorOffset,
            format: 'float32x4'
          }
        ],
      },
    ];
    const layout = this.device.createPipelineLayout({
        bindGroupLayouts : [this.uboBindGroupLayout,], });
    var desc = getPipelineDescriptor(this.device, layout, obstaclesWGSL, 'vs_main', 'fs_main', 
      buffers, presentationFormat, 'triangle-list', 'back');
    this.obstaclesPipeline = this.device.createRenderPipeline(desc);
  }

  initGoalPipeline(presentationFormat, cbm : ComputeBufferManager) {
    const buffers = [
      {
        arrayStride: 6*4,
        stepMode: 'instance',
        attributes: [
          {
            // position
            shaderLocation: 0,
            offset: 0,
            format: 'float32x4',
          },
        ],
      },
      {
        arrayStride: sphereItemSize,
        attributes: [
          {
            // position
            shaderLocation: 1,
            offset: spherePosOffset,
            format: 'float32x4',
          },
          {
            // normal
            shaderLocation: 2,
            offset: sphereNorOffset,
            format: 'float32x4',
          }
        ],
      },
    ];
    const layout = this.device.createPipelineLayout({
        bindGroupLayouts : [this.uboBindGroupLayout,], });
    var desc = getPipelineDescriptor(this.device, layout, renderWGSL, 'vs_goal', 'fs_goal', 
      buffers, presentationFormat, 'line-list', 'back');
    this.goalPipeline = this.device.createRenderPipeline(desc);
  }

  drawPlatform(device: GPUDevice, passEncoder: GPURenderPassEncoder, platformWidth) {
    const modelMatrix = mat4.create();
    mat4.identity(modelMatrix);
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(platformWidth, 0.1, platformWidth));
    this.device.queue.writeBuffer(this.platformModelUBO, 0, modelMatrix as Float32Array);

    passEncoder.setPipeline(this.platformPipeline);
    passEncoder.setBindGroup(0, this.platformBindGroup);
    passEncoder.setBindGroup(1, this.platformModelBindGroup);
    passEncoder.setVertexBuffer(0, this.platformVertexBuffer);
    passEncoder.draw(platformVertexCount, 1, 0, 0);
  }

  drawCrowdShadow(device: GPUDevice, renderCmd: GPUCommandEncoder, agentsBuffer: GPUBuffer, numAgents: number) {
    const shadowPass = renderCmd.beginRenderPass(this.shadowPassDescriptor);
    shadowPass.setPipeline(this.crowdShadowPipeline);
    shadowPass.setBindGroup(0, this.crowdShadowBindGroup);
    shadowPass.setBindGroup(1, this.crowdModelBindGroup);
    shadowPass.setVertexBuffer(0, agentsBuffer);
    shadowPass.setVertexBuffer(1, this.meshVertexBuffer);
    shadowPass.draw(this.mesh.vertexCount, numAgents, 0, 0);
    shadowPass.endPass();
  }

  drawCrowd(device: GPUDevice, passEncoder: GPURenderPassEncoder, agentsBuffer: GPUBuffer, numAgents: number) {
    passEncoder.setPipeline(this.crowdPipeline);
    passEncoder.setBindGroup(0, this.crowdBindGroup);
    passEncoder.setBindGroup(1, this.crowdModelBindGroup);
    passEncoder.setVertexBuffer(0, agentsBuffer);
    passEncoder.setVertexBuffer(1, this.meshVertexBuffer);
    passEncoder.draw(this.mesh.vertexCount, numAgents, 0, 0);
  }
  
  drawObstacles(device: GPUDevice, passEncoder: GPURenderPassEncoder, obstaclesBuffer: GPUBuffer, numObstacles: number) {
    passEncoder.setPipeline(this.obstaclesPipeline);
    passEncoder.setBindGroup(0, this.obstaclesBindGroup);
    passEncoder.setVertexBuffer(0, obstaclesBuffer);
    passEncoder.setVertexBuffer(1, this.obstacleVertexBuffer);
    passEncoder.draw(cubeVertexCount, numObstacles, 0, 0);
  }

  drawGoals(device: GPUDevice, passEncoder: GPURenderPassEncoder, goalsBuffer: GPUBuffer, numGoals: number) {
    passEncoder.setPipeline(this.goalPipeline);
    passEncoder.setBindGroup(0, this.goalBindGroup);
    passEncoder.setVertexBuffer(0, goalsBuffer);
    passEncoder.setVertexBuffer(1, this.goalVertBuffer);
    passEncoder.draw(sphereVertCount, numGoals, 0, 0);
  } 

  updateSceneUBO(camera: Camera, gridOn: boolean, time: number){
    const vp = mat4.create();
    mat4.multiply(vp, camera.projectionMatrix, camera.viewMatrix);
    this.device.queue.writeBuffer(
      this.sceneUBO,
      // skip lightViewProj
      4 * 16,
      vp as Float32Array);
    this.device.queue.writeBuffer(
      this.sceneUBO,
      // lightViewProj, camViewProj, lightPos
      2 * 4 * 16 + 3 * 4,
      new Float32Array([gridOn ? 1.0 : 0.0]));
    this.device.queue.writeBuffer(
      this.sceneUBO,
      // lightViewProj, camViewProj, lightPos, gridOn
      2 * 4 * 16 + 4 * 4,
      new Float32Array([camera.controls.eye[0], camera.controls.eye[1], camera.controls.eye[2]]));
    this.device.queue.writeBuffer(
      this.sceneUBO,
      // lightViewProj, camViewProj, lightPos, gridOn, camPos
      2 * 4 * 16 + 4 * 4 + 3 * 4,
      new Float32Array([time]));
  }
}

////////////////////////////////////////////////////////////////////////////////////////
//                         renderBufferManager Helpers                                //
////////////////////////////////////////////////////////////////////////////////////////

// Create a vertex buffer from the given data
const createVBO = (device: GPUDevice, vertexArray: Float32Array) => {
  const vertexBuffer = device.createBuffer({
      size: vertexArray.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(vertexArray);
  vertexBuffer.unmap();
  return vertexBuffer;
}

const createUBO = (device: GPUDevice, size: number) => {
  return device.createBuffer({
    size: size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

const createBindGroup = (device: GPUDevice, bgl: GPUBindGroupLayout, uniformBuffer: GPUBuffer) => {
  const bg = device.createBindGroup({
    layout: bgl,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });
  return bg;
}

const createTexturedBindGroup = (device: GPUDevice, bgl: GPUBindGroupLayout, uniformBuffer: GPUBuffer, texView: GPUTextureView, sampler) => {
  return device.createBindGroup({
  layout: bgl,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer,
      },
    },
    {
      binding: 1,
      resource: sampler,
    },
    {
      binding: 2,
      resource: texView,
    }
  ],
  });
}

const getPipelineDescriptor = (device: GPUDevice, layout: GPUPipelineLayout, code, vs: string, fs: string, 
  vertexBuffers, presentationFormat: GPUTextureFormat, primitiveType: GPUPrimitiveTopology, cullMode: GPUCullMode) => {

  const descriptor : GPURenderPipelineDescriptor = {
    layout: layout,
    vertex: {
      module: device.createShaderModule({code: headerWGSL + code}),
      entryPoint: vs,
      buffers: vertexBuffers,
    },
    fragment : {
      module: device.createShaderModule({code: headerWGSL + code}),
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
      format: <GPUTextureFormat>'depth24plus-stencil8',
    },
  };
  return descriptor;

}

const getDepthTexture = (device: GPUDevice, presentationSize) => {
  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus-stencil8',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return depthTexture;
}