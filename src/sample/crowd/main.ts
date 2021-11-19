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
  cubeVertexArray,
  cubeVertexSize,
  cubeUVOffset,
  cubePositionOffset,
  cubeVertexCount,
} from '../../meshes/cube';

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
import { prototype } from 'module';

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
let resetSim : boolean;

// Reset camera to original settings (gui function)
function resetCameraFunc() {
  camera = new Camera(vec3.fromValues(3, 3, 3), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

const init: SampleInit = async ({ canvasRef, gui, stats }) => {

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
    resetCamera: resetCameraFunc,
    gridOn: true
  };

  let prevGridWidth = guiParams.gridWidth;
  resetSim = false;

  let gridFolder = gui.addFolder("Grid");
  gridFolder.add(guiParams, 'gridWidth', 1, 500, 1);
  gridFolder.add(guiParams, 'gridOn');
  gridFolder.open();
  let camFolder = gui.addFolder("Camera");
  camFolder.add(guiParams, 'resetCamera');
  camFolder.open();

  const simulationParams = {
    simulate: true,
    deltaTime: 0.04,
    numAgents: 100000,
    resetSimulation: () => { resetSim = true; }
  };

  let prevNumAgents = simulationParams.numAgents;

  let simFolder = gui.addFolder("Simulation");
  simFolder.add(simulationParams, 'simulate');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.01);
  simFolder.add(simulationParams, 'numAgents', 1000, 100000, 10);
  simFolder.add(simulationParams, 'resetSimulation');
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

  let initialAgentData = getAgentData(simulationParams.numAgents);
  let agentsBuffer = device.createBuffer({
    size: simulationParams.numAgents * agentInstanceByteSize,
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
  var renderPipelineCrowd = getCrowdRenderPipeline(
        device, crowdWGSL, agentInstanceByteSize, agentPositionOffset, 
        agentColorOffset, cubeVertexSize, cubePositionOffset, cubeUVOffset, presentationFormat
  );

  // Get the depth texture for both pipelines
  const depthTexture = getDepthTexture(device, presentationSize);

  const uniformBufferPlatform = getUniformBuffer(device, 4 * 16);
  const uniformBufferGridLines = getUniformBuffer(device, 4 * 16);
  
  const uniformBindGroupPlatform = getUniformBindGroup(device, pipelinePlatform, uniformBufferPlatform);
  const uniformBindGroupGridLines = getUniformBindGroup(device, pipelineGridLines, uniformBufferGridLines);

  const uniformBufferCrowd = getUniformBuffer(device, 4 * 16);
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
  // Prototype buffer
  //////////////////////////////////////////////////////////////////////////////
  const prototypeVertexCount = cubeVertexCount;
  const prototypeVerticesBuffer = getVerticesBuffer(device, cubeVertexArray);

  //////////////////////////////////////////////////////////////////////////////
  // Simulation compute pipeline
  //////////////////////////////////////////////////////////////////////////////

  let simulationUBOBufferSize =
    1 * 4 + // deltaTime
    3 * 4 + // padding
    4 * 4 + // seed
    0;
  let simulationUBOBuffer = device.createBuffer({
    size: simulationUBOBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let computePipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({
        code: crowdWGSL,
      }),
      entryPoint: 'simulate',
    },
  });
  let computeBindGroup = device.createBindGroup({
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
          size: simulationParams.numAgents * agentInstanceByteSize,
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
    stats.begin();
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
      if (prevNumAgents != simulationParams.numAgents) {
        prevNumAgents = simulationParams.numAgents;
        // set reset sim to true so that simulation starts over
        // and agents are redistributed
        resetSim = true;
      }
      // recompute agent buffer if resetSim button pressed
      if (resetSim){
        initialAgentData = getAgentData(simulationParams.numAgents);
        agentsBuffer = device.createBuffer({
          size: simulationParams.numAgents * agentInstanceByteSize,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
          mappedAtCreation: true
        });
        new Float32Array(agentsBuffer.getMappedRange()).set(
          initialAgentData
        );
        agentsBuffer.unmap();
        computeBindGroup = device.createBindGroup({
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
                size: simulationParams.numAgents * agentInstanceByteSize,
              },
            },
          ],
        });
        resetSim = false;
      }

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
      passEncoder.dispatch(Math.ceil(simulationParams.numAgents / 64));
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
      if (guiParams.gridOn){
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
      }

      // -------------- Draw Crowds ------------------------ // 
      let mvp = getCrowdTransform();
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
        ])
      );
      passEncoder.setPipeline(renderPipelineCrowd);
      passEncoder.setBindGroup(0, uniformBindGroupCrowd);
      passEncoder.setVertexBuffer(0, agentsBuffer);
      passEncoder.setVertexBuffer(1, prototypeVerticesBuffer);
      passEncoder.draw(prototypeVertexCount, simulationParams.numAgents, 0, 0);
      passEncoder.endPass();
    }

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
    stats.end();
  }
  requestAnimationFrame(frame);
};

const Crowd: () => JSX.Element = () =>
  makeSample({
    name: 'Crowd Simulation',
    description:
      'This is a WebGPU Crowd Simulation.',
    init,
    gui: true,
    stats: true,
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