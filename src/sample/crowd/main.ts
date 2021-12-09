import { mat4, vec3, vec4 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import Camera from "./Camera";

import { TestScene, ComputeBufferManager } from './crowdUtils';
import { RenderBufferManager } from './renderUtils';

import renderWGSL from '../../shaders/background.render.wgsl';
import crowdWGSL from '../../shaders/crowd.render.wgsl';
import explicitIntegrationWGSL from '../../shaders/explicitIntegration.compute.wgsl';
import assignCellsWGSL from '../../shaders/assignCells.compute.wgsl';
import buildHashGrid from '../../shaders/buildHashGrid.compute.wgsl';
import contactSolveWGSL from '../../shaders/contactSolve.compute.wgsl';
import constraintSolveWGSL from '../../shaders/constraintSolve.compute.wgsl';
import finalizeVelocityWGSL from '../../shaders/finalizeVelocity.compute.wgsl';
import headerWGSL from '../../shaders/header.compute.wgsl';

import {loadModel, Mesh} from "../../meshes/mesh";
import { meshDictionary } from './meshDictionary';

let camera : Camera;
let aspect : number;
let resetSim : boolean;

// Reset camera to original settings (gui function)
function resetCameraFunc(x: number = 50, y: number = 50, z: number = 50) {
  camera = new Camera(vec3.fromValues(x, y, z), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

function getSortStepWGSL(numAgents : number, k : number, j : number, ){
  // bitonic sort requires a device-wide join after every "step" to avoid
  // race conditions. The least gross way I can think to do that is to create a new pipeline
  // for each step.
  let baseWGSL = `
  [[binding(1), group(0)]] var<storage, read_write> agentData : Agents;

  fn swap(idx1 : u32, idx2 : u32) {
    var tmp = agentData.agents[idx1];
    agentData.agents[idx1] = agentData.agents[idx2];
    agentData.agents[idx2] = tmp; 
  }

  fn agentlt(idx1 : u32, idx2 : u32) -> bool {
    return agentData.agents[idx1].cell < agentData.agents[idx2].cell;
  }

  fn agentgt(idx1 : u32, idx2 : u32) -> bool {
    return agentData.agents[idx1].cell > agentData.agents[idx2].cell;
  }

  [[stage(compute), workgroup_size(256)]]
  fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let idx = GlobalInvocationID.x ;
    
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
                code: headerWGSL + getSortStepWGSL(numAgents, k, j),
              }),
              entryPoint: 'main',
            },
          })
        );
      }
    }
}

const init: SampleInit = async ({ canvasRef, gui, stats }) => {

  canvasRef.current.width = document.body.clientWidth - 50;
  canvasRef.current.height = canvasRef.current.width * (800 / 1600);

  ///////////////////////////////////////////////////////////////////////
  //                       Camera Setup                                //
  ///////////////////////////////////////////////////////////////////////

  camera = new Camera(vec3.fromValues(50, 50, 50), vec3.fromValues(0, 0, 0));
  aspect = canvasRef.current.width / canvasRef.current.height;
  //console.log(document.body.clientWidth);
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();

  ////////////////////////////////////////////////////////////////////////
  //                        GUI Setup                                   //
  ////////////////////////////////////////////////////////////////////////
  
  const guiParams = {
    gridWidth: 200,
    resetCamera: resetCameraFunc,
    gridOn: true
  };

  let prevGridWidth = guiParams.gridWidth;
  resetSim = true;

  let gridFolder = gui.addFolder("Grid");
  gridFolder.add(guiParams, 'gridWidth', 1, 5000, 1);
  gridFolder.add(guiParams, 'gridOn');
  gridFolder.open();
  let camFolder = gui.addFolder("Camera");
  camFolder.add(guiParams, 'resetCamera');
  camFolder.open();

  const simulationParams = {
    simulate: true,
    deltaTime: 0.02,
    numAgents: 1024,
    numObstacles : 1,
    avoidance: false,
    gridWidth: guiParams.gridWidth,
    testScene: TestScene.PROXIMAL,
    resetSimulation: () => { resetSim = true; }
  };

  let prevNumAgents = simulationParams.numAgents;
  let prevTestScene = TestScene.DENSE;

  let simFolder = gui.addFolder("Simulation");
  simFolder.add(simulationParams, 'simulate');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.0001);
  simFolder.add(simulationParams, 'numAgents', 10, 100000, 2);
  simFolder.add(simulationParams, 'avoidance');
  simFolder.add(simulationParams, 'testScene', {
    'Proximal Behavior': TestScene.PROXIMAL, 
    'Bottleneck': TestScene.BOTTLENECK,
    'Dense Passing': TestScene.DENSE,
    'Sparse Passing': TestScene.SPARSE,
    'Obstacles': TestScene.OBSTACLES,
  });
  simFolder.add(simulationParams, 'resetSimulation');
  simFolder.open();

  const modelParams = {
    model: 'Duck'
  }
  let prevModel = 'Duck';
  gui.add(modelParams, 'model', ['Archer', 'Duck', 'Cesium Man']);


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
                                                 simulationParams.testScene,
                                                 simulationParams.numAgents,
                                                 simulationParams.gridWidth);

  //////////////////////////////////////////////////////////////////////////
  //                Render Buffer and Pipeline Setup                      //
  //////////////////////////////////////////////////////////////////////////
  var renderBuffManager : RenderBufferManager;

  let gridTexture: GPUTexture;
  {
    const img = document.createElement('img');
    img.src = require('../../../assets/img/checkerboard.png');
    await img.decode();
    const imageBitmap = await createImageBitmap(img);

    gridTexture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: gridTexture },
      [imageBitmap.width, imageBitmap.height]
    );
  }

  var bufManagerExists = false;
  var modelData = meshDictionary[modelParams.model];
  loadModel(modelData.filename).then((mesh : Mesh) => {
    mesh.scale = modelData.scale;
    renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
      presentationFormat, presentationSize,
      compBuffManager, mesh, gridTexture, sampler);
    
    bufManagerExists = true;
  });

  // Create a sampler with linear filtering for smooth interpolation.
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  //////////////////////////////////////////////////////////////////////////////
  // Create Compute Pipelines
  //////////////////////////////////////////////////////////////////////////////
  {
    var computeShadersPreSort = [
      headerWGSL + explicitIntegrationWGSL, 
      headerWGSL + assignCellsWGSL,
    ];
    var computeShadersPostSort = [
      headerWGSL + buildHashGrid,
      headerWGSL + contactSolveWGSL, 
      headerWGSL + constraintSolveWGSL, 
      headerWGSL + finalizeVelocityWGSL
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
                         compBuffManager.numAgents, 
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

  function getViewProjection() {
    const modelViewProjectionMatrix = mat4.create();

    mat4.multiply(modelViewProjectionMatrix, camera.projectionMatrix, camera.viewMatrix);
    return modelViewProjectionMatrix as Float32Array;
  }
  
  function frame() {
    stats.begin();
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    canvasRef.current.width = document.body.clientWidth - 50;
    canvasRef.current.height = canvasRef.current.width * (800 / 1600);

    // Compute new grid lines if there's a change in the gui
    if (prevGridWidth != guiParams.gridWidth) {
      renderBuffManager.resetGridLinesBuffer(guiParams.gridWidth);
      resetSim = true;
      simulationParams.gridWidth = guiParams.gridWidth;
      prevGridWidth = guiParams.gridWidth;
    }

    if (prevModel != modelParams.model) {
      bufManagerExists = false;
      prevModel = modelParams.model;
      var modelData = meshDictionary[modelParams.model];
      loadModel(modelData.filename).then((mesh : Mesh) => {
        mesh.scale = modelData.scale;
        renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
          presentationFormat, presentationSize,
          compBuffManager, mesh, gridTexture, sampler);
    
        bufManagerExists = true;
      });
    }

    camera.update();

    //------------------ Compute Calls ------------------------ //
    {
      if (prevNumAgents != simulationParams.numAgents) {
        // NOTE: we also reset the sim if the grid width changes
        // which is checked just above this
        prevNumAgents = simulationParams.numAgents;
        // set reset sim to true so that simulation starts over
        // and agents are redistributed
        resetSim = true;
      }

      if (prevTestScene != simulationParams.testScene) {
        prevTestScene = simulationParams.testScene;
        switch(simulationParams.testScene) {
          case TestScene.PROXIMAL:
            resetCameraFunc(5,10,5);
            compBuffManager.numValidAgents = 1<<6;
            simulationParams.numObstacles = 0;
            guiParams.resetCamera = () => resetCameraFunc(5, 10, 5);
            break;
          case TestScene.BOTTLENECK:
            resetCameraFunc(50,50,50);
            compBuffManager.numValidAgents = 1<<10;
            simulationParams.numObstacles = 2;
            guiParams.resetCamera = () => resetCameraFunc(50, 50, 50);
            break;
          case TestScene.DENSE:
            resetCameraFunc(50,50,50);
            compBuffManager.numValidAgents = 1<<15;
            simulationParams.numObstacles = 0;
            guiParams.resetCamera = () => resetCameraFunc(50, 50, 50);
            break;
          case TestScene.SPARSE:
            resetCameraFunc(50,50,50);
            compBuffManager.numValidAgents = 1<<12;
            simulationParams.numObstacles = 0;
            guiParams.resetCamera = () => resetCameraFunc(50, 50, 50);
            break;
          case TestScene.OBSTACLES:
            resetCameraFunc(50,50,50);
            compBuffManager.numValidAgents = 1<<10;
            simulationParams.numObstacles = 5;
            guiParams.resetCamera = () => resetCameraFunc(50, 50, 50);
            break;
        }
        resetSim = true;
      }

      // recompute agent buffer if resetSim button pressed
      if (resetSim) {
        compBuffManager.testScene = simulationParams.testScene;
        //compBuffManager.numValidAgents = simulationParams.numAgents;
        compBuffManager.gridWidth = simulationParams.gridWidth;

        // NOTE: Can't have 0 binding size so we just set to 1 dummy if no obstacles
        compBuffManager.numObstacles = Math.max(simulationParams.numObstacles, 1);

        // reinitilize buffers based on the new number of agents
        compBuffManager.initBuffers();
        computeBindGroup = compBuffManager.getBindGroup();
        // the number of steps in the sort pipeline is proportional
        // to log2 the number of agents, so reinitiliaze it
        fillSortPipelineList(device, 
                            compBuffManager.numAgents, 
                            computePipelinesSort, 
                            compBuffManager);
        resetSim = false;
      }

      var computeCommand = device.createCommandEncoder();

      if (simulationParams.simulate) {
      // write the parameters to the Uniform buffer for our compute shaders
      compBuffManager.writeSimParams(simulationParams);

      // execute each compute shader in the order they were pushed onto
      // the computePipelines array
      var passEncoder = computeCommand.beginComputePass();
      //// ----- Compute Pass Before Sort -----
      for (let i = 0; i < computePipelinesPreSort.length; i++){
        passEncoder.setPipeline(computePipelinesPreSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(compBuffManager.numAgents / 64));
      }

      // ----- Compute Pass Sort -----
      for (let i = 0; i < computePipelinesSort.length; i++){
        passEncoder.setPipeline(computePipelinesSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(compBuffManager.numAgents / 256));
      }
      
      // ----- Compute Pass Post Sort 1 -----
      const constraintShaderIdx = 2;
      for (let i = 0; i < constraintShaderIdx; i++){
        passEncoder.setPipeline(computePipelinesPostSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(compBuffManager.numAgents / 64));
      }
      passEncoder.endPass();

      device.queue.submit([computeCommand.finish()]);
    
      
      // ----- Compute Pass Constraint Solve -----
      // Since WebGPU does not support push constants, need to add write buffer to queue
      // SEE: https://github.com/gpuweb/gpuweb/issues/762#issuecomment-625622428
      for (let i = 0; i < 6; i++) {
        computeCommand = device.createCommandEncoder();
        compBuffManager.setIteration(i+1);  // Set iteration number for compute shader
        passEncoder = computeCommand.beginComputePass();
        passEncoder.setPipeline(computePipelinesPostSort[constraintShaderIdx]);
        passEncoder.setBindGroup(0, computeBindGroup);
        passEncoder.dispatch(Math.ceil(compBuffManager.numAgents / 64));
        passEncoder.endPass();
        device.queue.submit([computeCommand.finish()]);
      }      

      // ----- Compute Pass Post Sort 2 -----
      computeCommand = device.createCommandEncoder();
      passEncoder = computeCommand.beginComputePass();
      for (let i = constraintShaderIdx+1; i < computePipelinesPostSort.length; i++){
        passEncoder.setPipeline(computePipelinesPostSort[i]);
        passEncoder.setBindGroup(0, computeBindGroup);
        // kick off the compute shader
        passEncoder.dispatch(Math.ceil(compBuffManager.numAgents / 64));
      }
      passEncoder.endPass();

      device.queue.submit([computeCommand.finish()])
    }
    }

    // ------------------ Render Calls ------------------------- //
    if (bufManagerExists) {
      const renderCommand = device.createCommandEncoder();
      const transformationMatrix = getTransformationMatrix();

      renderBuffManager.renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();
      
      const passEncoder = renderCommand.beginRenderPass(renderBuffManager.renderPassDescriptor);

      // ----------------------- Draw ------------------------- //
      renderBuffManager.drawPlatform(device, transformationMatrix, passEncoder, guiParams.gridOn);
      //if (guiParams.gridOn)
      //  renderBuffManager.drawGridLines(device, transformationMatrix, passEncoder);

      const vp = getViewProjection();
      const camPos = vec3.fromValues(camera.controls.eye[0], camera.controls.eye[1], camera.controls.eye[2]) ;
      renderBuffManager.drawCrowd(device, vp, passEncoder, compBuffManager.agentsBuffer, compBuffManager.numAgents, camPos);

      if (simulationParams.numObstacles > 0)
        renderBuffManager.drawObstacles(device, vp, passEncoder, compBuffManager.obstaclesBuffer, compBuffManager.numObstacles);

      passEncoder.endPass();
      device.queue.submit([renderCommand.finish()]);
    }

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
        name: '../../shaders/background.render.wgsl',
        contents: renderWGSL,
        editable: true,
      },
      {
        name: '../../shaders/crowd.render.wgsl',
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
