import { mat4, vec3 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import Camera from "./Camera";

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

import basicVertWGSL from '../../shaders/basic.vert.wgsl';
import vertexPositionColorWGSL from '../../shaders/vertexPositionColor.frag.wgsl';
import gridLinesWGSL from './shaders.wgsl';

const init: SampleInit = async ({ canvasRef }) => {

  // create camera
  const camera = new Camera(vec3.fromValues(0, 2, 5), vec3.fromValues(0, 0, 0));

  const aspect = canvasRef.current.width / canvasRef.current.height;
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();

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

  // Create a vertex buffer from the cube data.
  const verticesBuffer = device.createBuffer({
    size: platformVertexArray.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(verticesBuffer.getMappedRange()).set(platformVertexArray);
  verticesBuffer.unmap();

  const pipeline = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: basicVertWGSL,
      }),
      entryPoint: 'main',
      buffers: [
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
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: vertexPositionColorWGSL,
      }),
      entryPoint: 'main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',

      // Backface culling since the cube is solid piece of geometry.
      // Faces pointing away from the camera will be occluded by faces
      // pointing toward the camera.
      cullMode: 'back',
    },

    // Enable depth testing so that the fragment closest to the camera
    // is rendered in front.
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });

  const depthTexture = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

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

  
  const gridLinesVertexArray = getGridLines(4);

  // Create a vertex buffer from the cube data.
  const verticesBuffer1 = device.createBuffer({
    size: gridLinesVertexArray.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(verticesBuffer1.getMappedRange()).set(gridLinesVertexArray);
  verticesBuffer1.unmap();

  const pipeline1 = device.createRenderPipeline({
    vertex: {
      module: device.createShaderModule({
        code: gridLinesWGSL,
      }),
      entryPoint: 'vs_gridLines',
      buffers: [
        {
          arrayStride: gridLinesVertexSize,
          attributes: [
            {
              // position
              shaderLocation: 0,
              offset: gridLinesPositionOffset,
              format: 'float32x4',
            },
            {
              // uv
              shaderLocation: 1,
              offset: gridLinesUVOffset,
              format: 'float32x2',
            },
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({
        code: gridLinesWGSL,
      }),
      entryPoint: 'fs_gridLines',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'line-list',
    },

    // Enable depth testing so that the fragment closest to the camera
    // is rendered in front.
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });

  const depthTexture1 = device.createTexture({
    size: presentationSize,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const uniformBufferSize1 = 4 * 16; // 4x4 matrix
  const uniformBuffer1 = device.createBuffer({
    size: uniformBufferSize1,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBindGroup1 = device.createBindGroup({
    layout: pipeline1.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer1,
        },
      },
    ],
  });

  const renderPassDescriptor1: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Assigned later

        loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        storeOp: 'store',
      },
    ],
    depthStencilAttachment: {
      view: depthTexture1.createView(),

      depthLoadValue: 1.0,
      depthStoreOp: 'store',
      stencilLoadValue: 0,
      stencilStoreOp: 'store',
    },
  };

  function getTransformationMatrix() {
    const modelMatrix = mat4.create();
    mat4.identity(modelMatrix);
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(2, 0.1, 2));

    //return modelViewProjectionMatrix as Float32Array;
    const modelViewProjectionMatrix = mat4.create();
    mat4.multiply(modelViewProjectionMatrix, camera.viewMatrix, modelMatrix);
    mat4.multiply(modelViewProjectionMatrix, camera.projectionMatrix, modelViewProjectionMatrix);
    return modelViewProjectionMatrix as Float32Array;
  }

  function frame() {
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    camera.update();
    const transformationMatrix = getTransformationMatrix();
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, verticesBuffer);
    passEncoder.draw(platformVertexCount, 1, 0, 0);
    //passEncoder.endPass();
    //device.queue.submit([commandEncoder.finish()]);

    device.queue.writeBuffer(
      uniformBuffer1,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    renderPassDescriptor1.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    //const commandEncoder1 = device.createCommandEncoder();
    //const passEncoder1 = commandEncoder.beginRenderPass(renderPassDescriptor1);
    passEncoder.setPipeline(pipeline1);
    passEncoder.setBindGroup(0, uniformBindGroup1);
    passEncoder.setVertexBuffer(0, verticesBuffer1);
    passEncoder.draw(gridLinesVertexCount, 1, 0, 0);
    passEncoder.endPass();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
};

const Scene: () => JSX.Element = () =>
  makeSample({
    name: 'Scene',
    description:
      'This is the default scene minus the crowd elements.',
    init,
    sources: [
      {
        name: __filename.substr(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: '../../shaders/basic.vert.wgsl',
        contents: basicVertWGSL,
        editable: true,
      },
      {
        name: '../../shaders/vertexPositionColor.frag.wgsl',
        contents: vertexPositionColorWGSL,
        editable: true,
      },
      {
        name: './shaders.wgsl',
        contents: gridLinesWGSL,
        editable: true,
      },
      {
        name: '../../meshes/platform.ts',
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        contents: require('!!raw-loader!../../meshes/platform.ts').default,
      },
      {
        name: '../../meshes/gridLines.ts',
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        contents: require('!!raw-loader!../../meshes/gridLines.ts').default,
      },
    ],
    filename: __filename,
  });

export default Scene;