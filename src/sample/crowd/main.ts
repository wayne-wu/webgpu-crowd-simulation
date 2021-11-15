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

import {
  getVerticesBuffer,
  getPipeline,
  getDepthTexture,
  getUniformBuffer,
  getUniformBindGroup,
  getCrowdRenderPipeline
} from './renderUtils';

import {
  getAgentData
} from './crowdUtils';

import renderWGSL from './shaders.wgsl';
import crowdWGSL from './crowd.wgsl';

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

let camera : Camera;
let aspect : number;

// Reset camera to original settings (gui function)
function resetCameraFunc() {
  camera = new Camera(vec3.fromValues(3, 3, 3), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

const init: SampleInit = async ({ canvasRef, gui }) => {

  //---------------------- Setup Camera --------------------------------//
  camera = new Camera(vec3.fromValues(3, 3, 3), vec3.fromValues(0, 0, 0));
  aspect = canvasRef.current.width / canvasRef.current.height;
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();

  ////////////////////////////////////////////////////////////////////////
  //                        GUI Setup                                   //
  ////////////////////////////////////////////////////////////////////////
  
  const guiParams = {
    gridWidth: 50,
    resetCamera: resetCameraFunc
  };

  let prevGridWidth = guiParams.gridWidth;
  let gridFolder = gui.addFolder("Grid");
  gridFolder.add(guiParams, 'gridWidth', 1, 500, 1);
  gridFolder.open();
  let camFolder = gui.addFolder("Camera");
  camFolder.add(guiParams, 'resetCamera');
  camFolder.open();

  const simulationParams = {
    simulate: true,
    deltaTime: 0.04,
  };

  let simFolder = gui.addFolder("Simulation");
  Object.keys(simulationParams).forEach((k) => {
    simFolder.add(simulationParams, k);
  });
  simFolder.open();


  /////////////////////////////////////////////////////////////////////////
  //                     Initial Context Setup                           //
  /////////////////////////////////////////////////////////////////////////

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

  ////////////////////////////////////////////////////////////////////////////
  //                   Render Pipelines Setup                               //
  ////////////////////////////////////////////////////////////////////////////

  // Create vertex buffers for the platform and the grid lines
  const verticesBufferPlatform = getVerticesBuffer(device, platformVertexArray);
  // Compute the grid lines based on an input gridWidth
  let gridLinesVertexArray = getGridLines(guiParams.gridWidth);
  let verticesBufferGridLines = getVerticesBuffer(device, gridLinesVertexArray);

  const initialAgentData = getAgentData(numAgents);
  const agentsBuffer = device.createBuffer({
    size: numAgents * agentInstanceByteSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  });
  new Float32Array(agentsBuffer.getMappedRange()).set(
    initialAgentData
  );
  agentsBuffer.unmap();

  // Create render pipelines for platform and grid lines
  const pipelinePlatform = getPipeline(
        device, renderWGSL, 'vs_main', 'fs_platform', platformVertexSize,
        platformPositionOffset, platformUVOffset, presentationFormat, 'triangle-list', 'back'
  );
  const pipelineGridLines = getPipeline(
        device, renderWGSL, 'vs_main', 'fs_gridLines', gridLinesVertexSize,
        gridLinesPositionOffset, gridLinesUVOffset, presentationFormat, 'line-list', 'none'
  );
  const renderPipelineCrowd = getCrowdRenderPipeline(
        device, crowdWGSL, agentInstanceByteSize, agentPositionOffset, 
        agentColorOffset, presentationFormat
  );

  // Get the depth texture for both pipelines
  const depthTexture = getDepthTexture(device, presentationSize);

  const uniformBufferPlatform = getUniformBuffer(device, 4 * 16);
  const uniformBufferGridLines = getUniformBuffer(device, 4 * 16);
  
  const uniformBindGroupPlatform = getUniformBindGroup(device, pipelinePlatform, uniformBufferPlatform);
  const uniformBindGroupGridLines = getUniformBindGroup(device, pipelineGridLines, uniformBufferGridLines);

  const uniformBufferSizeCrowd =
    4 * 4 * 4 + // modelViewProjectionMatrix : mat4x4<f32>
    3 * 4 + // right : vec3<f32>
    4 + // padding
    3 * 4 + // up : vec3<f32>
    4 + // padding
    0;
  const uniformBufferCrowd = getUniformBuffer(device, uniformBufferSizeCrowd);
  const uniformBindGroupCrowd = getUniformBindGroup(device, renderPipelineCrowd, uniformBufferCrowd);

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
  // Simulation compute pipeline
  //////////////////////////////////////////////////////////////////////////////

  const simulationUBOBufferSize =
    1 * 4 + // deltaTime
    3 * 4 + // padding
    4 * 4 + // seed
    0;
  const simulationUBOBuffer = device.createBuffer({
    size: simulationUBOBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

  function getTransformationMatrix() {
    const modelMatrix = mat4.create();
    mat4.identity(modelMatrix);
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(50, 0.1, 50));

    const modelViewProjectionMatrix = mat4.create();
    mat4.multiply(modelViewProjectionMatrix, camera.viewMatrix, modelMatrix);
    mat4.multiply(modelViewProjectionMatrix, camera.projectionMatrix, modelViewProjectionMatrix);
    return modelViewProjectionMatrix as Float32Array;
  }

  function getCrowdTransform() {
    const modelViewProjectionMatrix = mat4.create();
    mat4.multiply(modelViewProjectionMatrix, camera.projectionMatrix, camera.viewMatrix);
    return modelViewProjectionMatrix;
  }

  function frame() {
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    // Compute new grid lines if there's a change in the gui
    if (prevGridWidth != guiParams.gridWidth) {
      gridLinesVertexArray = getGridLines(guiParams.gridWidth);
      verticesBufferGridLines = getVerticesBuffer(device, gridLinesVertexArray);
      prevGridWidth = guiParams.gridWidth;
    }

    camera.update();

    const commandEncoder = device.createCommandEncoder();

    //------------------ Compute Calls ------------------------ //
    {
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

      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, computeBindGroup);
      passEncoder.dispatch(Math.ceil(numAgents / 64));
      passEncoder.endPass();
    }
    // ------------------ Render Calls ------------------------- //
    {
      const transformationMatrix = getTransformationMatrix();

      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

      // ------------- Draw Platform ---------------------- //
      device.queue.writeBuffer(
        uniformBufferPlatform,
        0,
        transformationMatrix.buffer,
        transformationMatrix.byteOffset,
        transformationMatrix.byteLength
      );
      passEncoder.setPipeline(pipelinePlatform);
      passEncoder.setBindGroup(0, uniformBindGroupPlatform);
      passEncoder.setVertexBuffer(0, verticesBufferPlatform);
      passEncoder.draw(platformVertexCount, 1, 0, 0);

      // ------------- Draw Grid Lines --------------------- //
      device.queue.writeBuffer(
        uniformBufferGridLines,
        0,
        transformationMatrix.buffer,
        transformationMatrix.byteOffset,
        transformationMatrix.byteLength
      );
      passEncoder.setPipeline(pipelineGridLines);
      passEncoder.setBindGroup(0, uniformBindGroupGridLines);
      passEncoder.setVertexBuffer(0, verticesBufferGridLines);
      passEncoder.draw(gridLinesVertexCount, 1, 0, 0);

      // -------------- Draw Crowds ------------------------ // 
      let mvp = getCrowdTransform();
      let view = camera.viewMatrix;
      // prettier-ignore
      device.queue.writeBuffer(
        uniformBufferCrowd,
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
      passEncoder.setPipeline(renderPipelineCrowd);
      passEncoder.setBindGroup(0, uniformBindGroupCrowd);
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

const Crowd: () => JSX.Element = () =>
  makeSample({
    name: 'Scene',
    description:
      'This is the default scene minus the crowd elements.',
    init,
    gui: true,
    sources: [
      {
        name: __filename.substr(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: './shaders.wgsl',
        contents: renderWGSL,
        editable: true,
      },
      {
        name: './crowd.wgsl',
        contents: crowdWGSL,
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

export default Crowd;