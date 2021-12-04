import { mat4, vec3 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import Camera from "./Camera";

import { ComputeBufferManager } from './crowdUtils';
import { renderBufferManager } from './renderUtils';

import renderWGSL from './shaders.wgsl';
import crowdWGSL from './crowd.wgsl';
import explicitIntegrationWGSL from '../../shaders/explicitIntegration.compute.wgsl';
import assignCellsWGSL from '../../shaders/assignCells.compute.wgsl';
import findNeighborsWGSL from '../../shaders/findNeighbors.compute.wgsl';
import contactSolveWGSL from '../../shaders/contactSolve.compute.wgsl';
import constraintSolveWGSL from '../../shaders/constraintSolve.compute.wgsl';
import finalizeVelocityWGSL from '../../shaders/finalizeVelocity.compute.wgsl';

import {loadModel, Mesh} from "../../meshes/mesh";

let camera : Camera;
let aspect : number;
let resetSim : boolean;

// Reset camera to original settings (gui function)
function resetCameraFunc() {
  camera = new Camera(vec3.fromValues(50, 50, 50), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}


function getSortStepWGSL(numAgents : number, k : number, j : number, ){
  // bitonic sort as I understand it requires a device-wide join after every "step" to avoid
  // race conditions. The least gross way I can think to do that is to create a new pipeline
  // for each step.
  let baseWGSL = `
  struct Agent {
    x  : vec3<f32>;  // position + radius
    r  : f32;
    c  : vec4<f32>;  // color
    v  : vec3<f32>;  // velocity + inverse mass
    w  : f32;
    xp : vec3<f32>;  // planned/predicted position
    goal: vec3<f32>;
    cell : u32;      // grid cell (linear form)
    nearNeighbors : array<u32, 20>; 
    farNeighbors : array<u32, 20>;
  };

  [[block]] struct Agents {
    agents : array<Agent>;
  };

  [[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

  fn swap(idx1 : u32, idx2 : u32) {
    //var tmp = agentData.agents[idx1];
    //agentData.agents[idx1] = agentData.agents[idx2];
    //agentData.agents[idx2] = tmp; 
    var tmp = agentData.agents[idx1].c[0];
    agentData.agents[idx1].c[0] = agentData.agents[idx2].c[0];
    agentData.agents[idx2].c[0] = tmp; 
  }

  fn agentlt(idx1 : u32, idx2 : u32) -> bool {
    //return agentData.agents[idx1].cell < agentData.agents[idx2].cell;
    return agentData.agents[idx1].c[0] < agentData.agents[idx2].c[0];
  }

  fn agentgt(idx1 : u32, idx2 : u32) -> bool {
    //return agentData.agents[idx1].cell > agentData.agents[idx2].cell;
    return agentData.agents[idx1].c[0] > agentData.agents[idx2].c[0];
  }

  [[stage(compute), workgroup_size(256)]]
  fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x ;
    var agent = agentData.agents[idx];
    
    var j : u32 = ${j}u;
    var k : u32 = ${k}u;
    var l = idx ^ j; 
    if (l > idx){
      if (  (idx & k) == 0u && agentgt(idx,l) || (idx & k) != 0u && agentlt(idx, l)){
        swap(idx, l);
      }
    }
  }`;

  // minify the wgsl
  return baseWGSL.replace('/\s+/g', ' ').trim();
}


function fillSortPipelineList(device,
                              numAgents : number, 
                              computePipelinesSort,
                              compBuffManager){

    // be sure the list is empty before pushing new pipelines
    computePipelinesSort.length = 0;

    // set up sort pipelines
    for (let k = 2; k <= numAgents; k *= 2){ // k is doubled every iteration
      for (let j = k/2; j > 0; j = Math.floor(j/2)){ // j is halved at every iteration, with truncation of fractional parts
        computePipelinesSort.push(
          device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [compBuffManager.bindGroupLayout]
            }),
            compute: {
              module: device.createShaderModule({
                code: getSortStepWGSL(numAgents, k, j),
              }),
              entryPoint: 'main',
            },
          })
        );
      }
    }
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
    numAgents: 1024,
    avoidance: false,
    resetSimulation: () => { resetSim = true; }
  };

  let prevNumAgents = simulationParams.numAgents;

  let simFolder = gui.addFolder("Simulation");
  simFolder.add(simulationParams, 'simulate');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.0001);
  simFolder.add(simulationParams, 'numAgents', 10, 100000, 10);
  simFolder.add(simulationParams, 'avoidance');
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
  var renderBuffManager : renderBufferManager;

  var bufManagerExists = false;
  loadModel('Duck.glb').then((mesh : Mesh) => {
    renderBuffManager = new renderBufferManager(device, guiParams.gridWidth, 
      presentationFormat, presentationSize,
      compBuffManager.agentInstanceSize,
      compBuffManager.agentPositionOffset, 
      compBuffManager.agentColorOffset, compBuffManager.agentVelocityOffset, mesh);
    
    bufManagerExists = true;
  });
  

  //////////////////////////////////////////////////////////////////////////////
  // Create Compute Pipelines
  //////////////////////////////////////////////////////////////////////////////
  {
    var computeShadersPreSort = [
      explicitIntegrationWGSL, 
      assignCellsWGSL,
    ];
    var computeShadersPostSort = [
      findNeighborsWGSL, 
      contactSolveWGSL, 
      constraintSolveWGSL, 
      finalizeVelocityWGSL
    ];
    var computePipelinesPreSort = [];
    var computePipelinesSort = [];
    var computePipelinesPostSort = [];

    // set up pre-sort pipelines
    for(let i = 0; i < computeShadersPreSort.length; i++){
      computePipelinesPreSort.push( 
          device.createComputePipeline({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [compBuffManager.bindGroupLayout]
          }),
          compute: {
            module: device.createShaderModule({
              code: computeShadersPreSort[i],
            }),
            entryPoint: 'main',
          },
        })
      );
    }

    // set up sort pipelines
    fillSortPipelineList(device, 
                         simulationParams.numAgents, 
                         computePipelinesSort, 
                         compBuffManager);

    // set up post sort pipelines
    for(let i = 0; i < computeShadersPostSort.length; i++){
      computePipelinesPostSort.push( 
          device.createComputePipeline({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [compBuffManager.bindGroupLayout]
          }),
          compute: {
            module: device.createShaderModule({
              code: computeShadersPostSort[i],
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
        fillSortPipelineList(device, 
                            simulationParams.numAgents, 
                            computePipelinesSort, 
                            compBuffManager);
        resetSim = false;
      }

      // write the parameters to the Uniform buffer for our compute shaders
      compBuffManager.writeSimParams(simulationParams);

      // execute each compute shader in the order they were pushed onto
      // the computePipelines array
      const passEncoder = commandEncoder.beginComputePass();
      // ----- Compute Pass Before Sort -----
      for (let i = 0; i < computePipelinesPreSort.length; i++){
        passEncoder.setPipeline(computePipelinesPreSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(simulationParams.numAgents / 64));
      }

      // ----- Compute Pass Sort -----
      for (let i = 0; i < computePipelinesSort.length; i++){
        passEncoder.setPipeline(computePipelinesSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(simulationParams.numAgents / 256));
      }

      // ----- Compute Pass After Sort -----
      for (let i = 0; i < computePipelinesPostSort.length; i++){
        passEncoder.setPipeline(computePipelinesPostSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(simulationParams.numAgents / 64));
      }
      passEncoder.endPass();

    }
    // ------------------ Render Calls ------------------------- //
    if (bufManagerExists) {
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
      {
        name: '../../meshes/mesh.ts',
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        contents: require('!!raw-loader!../../meshes/mesh.ts').default,
      },
    ],
    filename: __filename,
  });

export default Crowd;
