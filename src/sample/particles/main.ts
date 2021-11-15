import { mat4, vec3 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';

import crowdWGSL from './crowd.wgsl';
import probabilityMapWGSL from './probabilityMap.wgsl';

const numAgents = 100000;
const agentPositionOffset = 0;
const agentColorOffset = 4 * 4;
const agentInstanceByteSize =
  3 * 4 + // position
  1 * 4 + // lifetime
  4 * 4 + // color
  3 * 4 + // velocity
  1 * 4 + // padding
  0;

const init: SampleInit = async ({ canvasRef, gui }) => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  if (canvasRef.current === null) return;
  const context = canvasRef.current.getContext('webgpu');

  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [
    canvasRef.current.clientWidth * devicePixelRatio,
    canvasRef.current.clientHeight * devicePixelRatio,
  ];
  const presentationFormat = context.getPreferredFormat(adapter);

  context.configure({
    device,
    format: presentationFormat,
    size: presentationSize,
  });

  const agentsBuffer = device.createBuffer({
    size: numAgents * agentInstanceByteSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  });

  const agentIdxOffset = 12;
  const initialAgentData = new Float32Array(numAgents * agentIdxOffset);  //48 is total byte size of each agent

  for (let i = 0; i < numAgents/2; ++i) {
    // position.xyz
    initialAgentData[agentIdxOffset * i + 0] = 10 * (Math.random() - 0.5);
    initialAgentData[agentIdxOffset * i + 1] = 0;
    initialAgentData[agentIdxOffset * i + 2] = 10 + 2 * (Math.random() - 0.5);

    // color.rgba
    initialAgentData[agentIdxOffset * i + 4] = 1;
    initialAgentData[agentIdxOffset * i + 5] = 0;
    initialAgentData[agentIdxOffset * i + 6] = 0;
    initialAgentData[agentIdxOffset * i + 7] = 1;

    // velocity.xyz
    initialAgentData[agentIdxOffset * i + 8] = 0;
    initialAgentData[agentIdxOffset * i + 9] = 0;
    initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random())*-1;
  }
  for (let i = numAgents/2; i < numAgents; ++i) {
    // position.xyz
    initialAgentData[agentIdxOffset * i + 0] = 10 * (Math.random() - 0.5);
    initialAgentData[agentIdxOffset * i + 1] = 0;
    initialAgentData[agentIdxOffset * i + 2] = -10 + 2 * (Math.random() - 0.5);

    // color.rgba
    initialAgentData[agentIdxOffset * i + 4] = 0;
    initialAgentData[agentIdxOffset * i + 5] = 0;
    initialAgentData[agentIdxOffset * i + 6] = 1;
    initialAgentData[agentIdxOffset * i + 7] = 1;

    // velocity.xyz
    initialAgentData[agentIdxOffset * i + 8] = 0;
    initialAgentData[agentIdxOffset * i + 9] = 0;
    initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random());
  }

  new Float32Array(agentsBuffer.getMappedRange()).set(
    initialAgentData
  );
  agentsBuffer.unmap();

  const renderPipeline = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: crowdWGSL,
      }),
      entryPoint: 'vs_main',
      buffers: [
        {
          // instanced agents buffer
          arrayStride: agentInstanceByteSize,
          stepMode: 'instance',
          attributes: [
            {
              // position
              shaderLocation: 0,
              offset: agentPositionOffset,
              format: 'float32x3',
            },
            {
              // color
              shaderLocation: 1,
              offset: agentColorOffset,
              format: 'float32x4',
            },
          ],
        },
        {
          // quad vertex buffer
          arrayStride: 2 * 4, // vec2<f32>
          stepMode: 'vertex',
          attributes: [
            {
              // vertex positions
              shaderLocation: 2,
              offset: 0,
              format: 'float32x2',
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: crowdWGSL,
      }),
      entryPoint: 'fs_main',
      targets: [
        {
          format: presentationFormat,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'zero',
              dstFactor: 'one',
              operation: 'add',
            },
          },
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

  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const uniformBufferSize =
    4 * 4 * 4 + // modelViewProjectionMatrix : mat4x4<f32>
    3 * 4 + // right : vec3<f32>
    4 + // padding
    3 * 4 + // up : vec3<f32>
    4 + // padding
    0;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });

  const renderPassDescriptor: GPURenderPassDescriptor = {
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

  //////////////////////////////////////////////////////////////////////////////
  // Quad vertex buffer
  //////////////////////////////////////////////////////////////////////////////
  const quadVertexBuffer = device.createBuffer({
    size: 6 * 2 * 4, // 6x vec2<f32>
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(quadVertexBuffer.getMappedRange()).set(
    new Float32Array([
      -1.0,
      -1.0,
      +1.0,
      -1.0,
      -1.0,
      +1.0,
      -1.0,
      +1.0,
      +1.0,
      -1.0,
      +1.0,
      +1.0,
    ])
  );
  quadVertexBuffer.unmap();

  //////////////////////////////////////////////////////////////////////////////
  // Texture
  //////////////////////////////////////////////////////////////////////////////
  let texture: GPUTexture;
  let textureWidth = 1;
  let textureHeight = 1;
  let numMipLevels = 1;
  {
    const img = document.createElement('img');
    img.src = require('../../../assets/img/webgpu.png');
    await img.decode();
    const imageBitmap = await createImageBitmap(img);

    // Calculate number of mip levels required to generate the probability map
    while (
      textureWidth < imageBitmap.width ||
      textureHeight < imageBitmap.height
    ) {
      textureWidth *= 2;
      textureHeight *= 2;
      numMipLevels++;
    }
    texture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      mipLevelCount: numMipLevels,
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: texture },
      [imageBitmap.width, imageBitmap.height]
    );
  }

  //////////////////////////////////////////////////////////////////////////////
  // Probability map generation
  // The 0'th mip level of texture holds the color data and spawn-probability in
  // the alpha channel. The mip levels 1..N are generated to hold spawn
  // probabilities up to the top 1x1 mip level.
  //////////////////////////////////////////////////////////////////////////////
  {
    const probabilityMapImportLevelPipeline = device.createComputePipeline({
      compute: {
        module: device.createShaderModule({ code: probabilityMapWGSL }),
        entryPoint: 'import_level',
      },
    });
    const probabilityMapExportLevelPipeline = device.createComputePipeline({
      compute: {
        module: device.createShaderModule({ code: probabilityMapWGSL }),
        entryPoint: 'export_level',
      },
    });

    const probabilityMapUBOBufferSize =
      1 * 4 + // stride
      3 * 4 + // padding
      0;
    const probabilityMapUBOBuffer = device.createBuffer({
      size: probabilityMapUBOBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const buffer_a = device.createBuffer({
      size: textureWidth * textureHeight * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    const buffer_b = device.createBuffer({
      size: textureWidth * textureHeight * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(
      probabilityMapUBOBuffer,
      0,
      new Int32Array([textureWidth])
    );
    const commandEncoder = device.createCommandEncoder();
    for (let level = 0; level < numMipLevels; level++) {
      const levelWidth = textureWidth >> level;
      const levelHeight = textureHeight >> level;
      const pipeline =
        level == 0
          ? probabilityMapImportLevelPipeline.getBindGroupLayout(0)
          : probabilityMapExportLevelPipeline.getBindGroupLayout(0);
      const probabilityMapBindGroup = device.createBindGroup({
        layout: pipeline,
        entries: [
          {
            // ubo
            binding: 0,
            resource: { buffer: probabilityMapUBOBuffer },
          },
          {
            // buf_in
            binding: 1,
            resource: { buffer: level & 1 ? buffer_a : buffer_b },
          },
          {
            // buf_out
            binding: 2,
            resource: { buffer: level & 1 ? buffer_b : buffer_a },
          },
          {
            // tex_in / tex_out
            binding: 3,
            resource: texture.createView({
              format: 'rgba8unorm',
              dimension: '2d',
              baseMipLevel: level,
              mipLevelCount: 1,
            }),
          },
        ],
      });
      if (level == 0) {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(probabilityMapImportLevelPipeline);
        passEncoder.setBindGroup(0, probabilityMapBindGroup);
        passEncoder.dispatch(Math.ceil(levelWidth / 64), levelHeight);
        passEncoder.endPass();
      } else {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(probabilityMapExportLevelPipeline);
        passEncoder.setBindGroup(0, probabilityMapBindGroup);
        passEncoder.dispatch(Math.ceil(levelWidth / 64), levelHeight);
        passEncoder.endPass();
      }
    }
    device.queue.submit([commandEncoder.finish()]);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Simulation compute pipeline
  //////////////////////////////////////////////////////////////////////////////
  const simulationParams = {
    simulate: true,
    deltaTime: 0.04,
  };

  const simulationUBOBufferSize =
    1 * 4 + // deltaTime
    3 * 4 + // padding
    4 * 4 + // seed
    0;
  const simulationUBOBuffer = device.createBuffer({
    size: simulationUBOBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  Object.keys(simulationParams).forEach((k) => {
    gui.add(simulationParams, k);
  });

  const computePipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({
        code: crowdWGSL,
      }),
      entryPoint: 'simulate',
    },
  });
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: simulationUBOBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: agentsBuffer,
          offset: 0,
          size: numAgents * agentInstanceByteSize,
        },
      },
    ],
  });

  const aspect = presentationSize[0] / presentationSize[1];
  const projection = mat4.create();
  const view = mat4.create();
  const mvp = mat4.create();
  mat4.perspective(projection, (2 * Math.PI) / 5, aspect, 1, 100.0);

  function frame() {
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    device.queue.writeBuffer(
      simulationUBOBuffer,
      0,
      new Float32Array([
        simulationParams.simulate ? simulationParams.deltaTime : 0.0,
        0.0,
        0.0,
        0.0, // padding
        Math.random() * 100,
        Math.random() * 100, // seed.xy
        1 + Math.random(),
        1 + Math.random(), // seed.zw
      ])
    );

    mat4.lookAt(view, vec3.fromValues(3,3,3), vec3.fromValues(0,0,0), vec3.fromValues(0,1,0));
    mat4.multiply(mvp, projection, view);

    // prettier-ignore
    device.queue.writeBuffer(
      uniformBuffer,
      0,
        new Float32Array([
        // modelViewProjectionMatrix
        mvp[0],  mvp[1],  mvp[2],  mvp[3],
        mvp[4],  mvp[5],  mvp[6],  mvp[7],
        mvp[8],  mvp[9],  mvp[10], mvp[11],
        mvp[12], mvp[13], mvp[14], mvp[15],

        view[0], view[4], view[8], // right

        0, // padding

        view[1], view[5], view[9], // up

        0, // padding
      ])
    );
    const swapChainTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();

    const commandEncoder = device.createCommandEncoder();
    {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, computeBindGroup);
      passEncoder.dispatch(Math.ceil(numAgents / 64));
      passEncoder.endPass();
    }
    {
      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, agentsBuffer);
      passEncoder.setVertexBuffer(1, quadVertexBuffer);
      passEncoder.draw(6, numAgents, 0, 0);
      passEncoder.endPass();
    }

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
};

const Particles: () => JSX.Element = () =>
  makeSample({
    name: 'Crowd Simulation',
    description:
      'This example demonstrates crowd simulation with compute shaders.',
    gui: true,
    init,
    sources: [
      {
        name: __filename.substr(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: './crowd.wgsl',
        contents: crowdWGSL,
        editable: true,
      },
      {
        name: './probabilityMap.wgsl',
        contents: probabilityMapWGSL,
        editable: true,
      },
    ],
    filename: __filename,
  });

export default Particles;
