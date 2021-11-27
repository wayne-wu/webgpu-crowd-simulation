import { mat4, vec3 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import Camera from "./Camera";

import { ComputeBufferManager } from './crowdUtils';
import { renderBufferManager } from './renderUtils';

import renderWGSL from './shaders.wgsl';
import crowdWGSL from './crowd.wgsl';
import explicitIntegrationWGSL from '../../shaders/explicitIntegration.compute.wgsl';
import findNeighborsWGSL from '../../shaders/findNeighbors.compute.wgsl';
import contactSolveWGSL from '../../shaders/contactSolve.compute.wgsl';
import constraintSolveWGSL from '../../shaders/constraintSolve.compute.wgsl';
import finalizeVelocityWGSL from '../../shaders/finalizeVelocity.compute.wgsl';


let camera : Camera;
let aspect : number;
let resetSim : boolean;

// Reset camera to original settings (gui function)
function resetCameraFunc() {
  camera = new Camera(vec3.fromValues(50, 50, 50), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

const init: SampleInit = async ({ canvasRef, gui, stats }) => {

  ///////////////////////////////////////////////////////////////////////
  //                       Camera Setup                                //
  ///////////////////////////////////////////////////////////////////////

  camera = new Camera(vec3.fromValues(50, 50, 50), vec3.fromValues(0, 0, 0));
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
    deltaTime: 0.02,
    numAgents: 100,
    resetSimulation: () => { resetSim = true; }
  };

  let prevNumAgents = simulationParams.numAgents;

  let simFolder = gui.addFolder("Simulation");
  simFolder.add(simulationParams, 'simulate');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.0001);
  simFolder.add(simulationParams, 'numAgents', 10, 100000, 10);
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

  /////////////////////////////////////////////////////////////////////////
  //                     Compute Buffer Setup                            //
  /////////////////////////////////////////////////////////////////////////
  var compBuffManager = new ComputeBufferManager(device, 
                                                 simulationParams.numAgents);

  //////////////////////////////////////////////////////////////////////////
  //                Render Buffer and Pipeline Setup                      //
  //////////////////////////////////////////////////////////////////////////
  var renderBuffManager = new renderBufferManager(device, guiParams.gridWidth, 
                                                  presentationFormat, presentationSize,
                                                  compBuffManager.agentInstanceSize,
                                                  compBuffManager.agentPositionOffset, 
                                                  compBuffManager.agentColorOffset);

  //////////////////////////////////////////////////////////////////////////////
  // Create Compute Pipelines
  //////////////////////////////////////////////////////////////////////////////
  {
    var computeShaders = [
      explicitIntegrationWGSL, 
      // findNeighborsWGSL, 
      contactSolveWGSL, 
      //constraintSolveWGSL, 
      finalizeVelocityWGSL
    ];
    var computePipelines = [];

    for(let i = 0; i < computeShaders.length; i++){
      computePipelines.push( 
          device.createComputePipeline({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [compBuffManager.bindGroupLayout]
          }),
          compute: {
            module: device.createShaderModule({
              code: computeShaders[i],
            }),
            entryPoint: 'main',
          },
        })
      );
    }
  }

  // get compute bind group
  var computeBindGroup = compBuffManager.getBindGroup();

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
      renderBuffManager.resetGridLinesBuffer(guiParams.gridWidth);
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
        compBuffManager.numAgents = simulationParams.numAgents;
        // reinitilize buffers based on the new number of agents
        compBuffManager.initBuffers();
        computeBindGroup = compBuffManager.getBindGroup();
        resetSim = false;
      }

      // write the parameters to the Uniform buffer for our compute shaders
      compBuffManager.writeSimParams(simulationParams);

      // execute each compute shader in the order they were pushed onto
      // the computePipelines array
      const passEncoder = commandEncoder.beginComputePass();
      for (let i = 0; i < computePipelines.length; i++){
        passEncoder.setPipeline(computePipelines[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(simulationParams.numAgents / 64));
      }
      passEncoder.endPass();

    }
    // ------------------ Render Calls ------------------------- //
    {
      const transformationMatrix = getTransformationMatrix();

      renderBuffManager.renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const passEncoder = commandEncoder.beginRenderPass(renderBuffManager.renderPassDescriptor);

      // ----------------------- Draw ------------------------- //
      renderBuffManager.drawPlatform(device, transformationMatrix, passEncoder);
      renderBuffManager.drawGridLines(device, transformationMatrix, passEncoder, guiParams.gridOn);
      renderBuffManager.drawCrowd(device, getCrowdTransform(), passEncoder, compBuffManager.agentsBuffer, simulationParams.numAgents);

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
      {
        name: '../../meshes/cube.ts',
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        contents: require('!!raw-loader!../../meshes/cube.ts').default,
      },
    ],
    filename: __filename,
  });

export default Crowd;