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
import { cubeVertexArray, cubeVertexCount } from '../../meshes/cube';
import { render } from 'react-dom';

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
  const baseWGSL = `
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
  
  // GUI PARAMETERS ------------------------------------------------------
  const guiParams = {
    gridWidth: 200,
    resetCamera: resetCameraFunc,
    gridOn: true
  };

  const sceneParams = {
    scene: TestScene.PROXIMAL,
    model: 'Cube',
    showGoals: true,
    shadowOn: true,
    'total agents': "", // dummy, autofilled later
    '2^x agents': 10
  }

  const simulationParams = {
    simulate: true,
    deltaTime: 0.02,
    numObstacles : 1,
    avoidanceModel: false,
    lookAhead : 6.0,
    gridWidth: guiParams.gridWidth,
    resetSimulation: () => { resetSim = true; }
  };

  // GUI GLOBALS ------------------------------------------------------------
  let prevGridWidth = guiParams.gridWidth;
  let prevNumAgents = sceneParams['2^x agents'];
  let prevTestScene = TestScene.DENSE;
  let prevModel = 'Duck';
  // default don't display slider to select number of agents -- will re-add if scene requires
  let numAgentsSliderDisplayed = false;
  resetSim = true;

  // GUI ELEMENTS -----------------------------------------------------------
  const gridFolder = gui.addFolder("Grid");
  gridFolder.add(guiParams, 'gridWidth', 1, 5000, 1);
  gridFolder.add(guiParams, 'gridOn');
  gridFolder.open();

  const camFolder = gui.addFolder("Camera");
  camFolder.add(guiParams, 'resetCamera');
  camFolder.open();

  const models = Array.from(Object.keys(meshDictionary));
  models.push('Cube');

  const sceneFolder = gui.addFolder("Scene");
  sceneFolder.add(sceneParams, 'scene', Object.values(TestScene));
  sceneFolder.add(sceneParams, 'model', models);
  sceneFolder.add(sceneParams, 'showGoals');
  sceneFolder.add(sceneParams, 'shadowOn');
  sceneFolder.add(sceneParams, 'total agents');
  sceneFolder.open();
  
  const simFolder = gui.addFolder("Simulation");
  simFolder.add(simulationParams, 'simulate');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.0001);
  simFolder.add(simulationParams, 'lookAhead', 3.0, 15.0, 1.0);
  simFolder.add(simulationParams, 'avoidanceModel');
  simFolder.add(simulationParams, 'resetSimulation');
  simFolder.open();

  // manually set text for total number of agents 
  const totalAgentsDOM = gui.__folders["Scene"].__controllers[4].domElement;
  totalAgentsDOM.innerHTML = totalAgentsDOM.innerHTML.substr(0, totalAgentsDOM.innerHTML.length - 1) + "disabled=\"true\">";
  totalAgentsDOM.innerText = "1024";


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
  const compBuffManager = new ComputeBufferManager(device,
                                                 sceneParams.scene,
                                                 sceneParams['2^x agents'],
                                                 simulationParams.gridWidth);

  //////////////////////////////////////////////////////////////////////////
  //                Render Buffer and Pipeline Setup                      //
  //////////////////////////////////////////////////////////////////////////
  let renderBuffManager : RenderBufferManager;

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
  // Create a sampler with linear filtering for smooth interpolation.
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  var platformWidth = 50; // global for platform width, test scenes change this

  let bufManagerExists = false;
  if (sceneParams.model == 'Cube'){
    const mesh = new Mesh(Array.from(cubeVertexArray), cubeVertexCount);
    mesh.scale = 0.2;
    renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
      presentationFormat, presentationSize,
      compBuffManager, mesh, gridTexture, sampler, 
      sceneParams.showGoals);
    bufManagerExists = true;
  }
  else{
    const modelData = meshDictionary[sceneParams.model];
    loadModel(modelData.filename, device).then((mesh : Mesh) => {
      mesh.scale = modelData.scale;
      renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
        presentationFormat, presentationSize,
        compBuffManager, mesh, gridTexture, sampler, 
        sceneParams.showGoals);
      
      bufManagerExists = true;
    });
  }

  //////////////////////////////////////////////////////////////////////////////
  // Create Compute Pipelines
  //////////////////////////////////////////////////////////////////////////////
  {
    const computeShadersPreSort = [
      headerWGSL + explicitIntegrationWGSL, 
      headerWGSL + assignCellsWGSL,
    ];
    const computeShadersPostSort = [
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

  function setTestScene(camPos: vec3, displayAgentSlider: boolean, numAgents: number, 
                        scenePlatformWidth: number, numObstacles: number, shadowOn: boolean){
    resetCameraFunc(camPos[0], camPos[1], camPos[2]); // set scene's camera position
    compBuffManager.numValidAgents = 1<<numAgents;    // number of agents to use in simulation
    sceneParams['2^x agents'] = numAgents;  // number of agents displayed in GUI
    platformWidth = scenePlatformWidth;               // size of the platform
    simulationParams.numObstacles = numObstacles;     // number of obstacles (used in compBufferManager, not gui)
    sceneParams.shadowOn = shadowOn;                      // display shadows on chosen scene

    // if agent slider exists and this scene doesn't support it, remove
    if (!displayAgentSlider && numAgentsSliderDisplayed) {
      sceneFolder.remove(gui.__folders["Scene"].__controllers[5]);
      numAgentsSliderDisplayed = false;
    }
    // if agent slider is supported and it doesn't exist, add it
    else if (displayAgentSlider && !numAgentsSliderDisplayed) {
      sceneFolder.add(sceneParams, '2^x agents', 1, 20, 1).listen();
      numAgentsSliderDisplayed = true;
    }
    
    // set camera reset function used in GUI (reset to this scene's default camera position)
    guiParams.resetCamera = () => resetCameraFunc(camPos[0], camPos[1], camPos[2]);
  }

  // get compute bind group
  var computeBindGroup;
  var computeBindGroup1 = compBuffManager.getBindGroup(false);
  var computeBindGroup2 = compBuffManager.getBindGroup(true);

  var time = 0;
  function frame() {
    time++;
    stats.begin();
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    // Compute new grid lines if there's a change in the gui
    if (prevGridWidth != guiParams.gridWidth) {
      resetSim = true;
      simulationParams.gridWidth = guiParams.gridWidth;
      prevGridWidth = guiParams.gridWidth;
    }

    if (prevModel != sceneParams.model) {
      bufManagerExists = false;
      prevModel = sceneParams.model;
      if (sceneParams.model == 'Cube'){
        const mesh = new Mesh(Array.from(cubeVertexArray), cubeVertexCount);
        mesh.scale = 0.2;
        renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
          presentationFormat, presentationSize,
          compBuffManager, mesh, gridTexture, sampler, 
          sceneParams.showGoals);
        bufManagerExists = true;
      } else {
      var modelData = meshDictionary[sceneParams.model];
      loadModel(modelData.filename, device).then((mesh : Mesh) => {
        mesh.scale = modelData.scale;
        renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
          presentationFormat, presentationSize,
          compBuffManager, mesh, gridTexture, sampler, 
          sceneParams.showGoals);
    
        bufManagerExists = true;
      });
    }
    }

    camera.update();

    //------------------ Compute Calls ------------------------ //
    {
      if (prevNumAgents != sceneParams['2^x agents']) {
        // NOTE: we also reset the sim if the grid width changes
        // which is checked just above this
        prevNumAgents = sceneParams['2^x agents'];
        totalAgentsDOM.innerText = Math.pow(2, prevNumAgents) + "";
        // set reset sim to true so that simulation starts over
        // and agents are redistributed
        resetSim = true;
      }

      if (prevTestScene != sceneParams.scene) {
        prevTestScene = sceneParams.scene;
        switch(sceneParams.scene) {
          case TestScene.PROXIMAL:
            setTestScene(vec3.fromValues(5, 10, 5), false, 6, 30, 0, true);
            break;
          case TestScene.BOTTLENECK:
            setTestScene(vec3.fromValues(20, 20, 20), false, 9, 63, 2, true);
            break;
          case TestScene.DENSE:
            setTestScene(vec3.fromValues(80, 75, 0), true, 13, 300, 0, false);
            break;
          case TestScene.SPARSE:
            setTestScene(vec3.fromValues(50, 50, 50), true, 12, 100, 0, false);
            break;
          case TestScene.OBSTACLES:
            setTestScene(vec3.fromValues(50, 50, 50), false, 10, 50, 5, true);
            break;
          case TestScene.CIRCLE:
            setTestScene(vec3.fromValues(5, 20, 5), false, 6, 20, 0, true);
            break;
          case TestScene.DISPERSED:
            setTestScene(vec3.fromValues(0, 60, 0), false, 11, 100, 0, false);
            break;
        }
        resetSim = true;
      }

      // recompute agent buffer if resetSim button pressed
      if (resetSim) {
        compBuffManager.testScene = sceneParams.scene;
        compBuffManager.numValidAgents = 1<<sceneParams['2^x agents'];
        compBuffManager.gridWidth = simulationParams.gridWidth;

        // NOTE: Can't have 0 binding size so we just set to 1 dummy if no obstacles
        compBuffManager.numObstacles = Math.max(simulationParams.numObstacles, 1);

        // reinitilize buffers based on the new number of agents
        compBuffManager.initBuffers();

        computeBindGroup1 = compBuffManager.getBindGroup(false);  // READ agents1 WRITE agents2
        computeBindGroup2 = compBuffManager.getBindGroup(true);   // READ agents2 WRITE agents1
        
        // the number of steps in the sort pipeline is proportional
        // to log2 the number of agents, so reinitiliaze it
        fillSortPipelineList(device, 
                            compBuffManager.numAgents, 
                            computePipelinesSort, 
                            compBuffManager);
        resetSim = false;
      }

      if(simulationParams.simulate) {

      computeBindGroup = computeBindGroup1;
      var computeCommand = device.createCommandEncoder();
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
      
      
      // Stability/Contact solve will write to different buffer
      computeBindGroup = computeBindGroup2;

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

        // ping-pong buffers
        if(computeBindGroup == computeBindGroup1)
          computeBindGroup = computeBindGroup2;
        else if(computeBindGroup == computeBindGroup2)
          computeBindGroup = computeBindGroup1;
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

      renderBuffManager.updateSceneUBO(camera, guiParams.gridOn, time, sceneParams.shadowOn);

      const renderCommand = device.createCommandEncoder();
      
      const agentsBuffer : GPUBuffer = computeBindGroup == computeBindGroup2 ? compBuffManager.agents1Buffer : compBuffManager.agents2Buffer;

      if(sceneParams.shadowOn)
        renderBuffManager.drawCrowdShadow(device, renderCommand, agentsBuffer, compBuffManager.numAgents);

      // const transformationMatrix = getTransformationMatrix();
      renderBuffManager.renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();
      
      const renderPass = renderCommand.beginRenderPass(renderBuffManager.renderPassDescriptor);

      // ----------------------- Draw ------------------------- //
      renderBuffManager.drawPlatform(device, renderPass, platformWidth);
      
      renderBuffManager.drawCrowd(device, renderPass, agentsBuffer, compBuffManager.numAgents);

      if (simulationParams.numObstacles > 0)
        renderBuffManager.drawObstacles(device, renderPass, compBuffManager.obstaclesBuffer, compBuffManager.numObstacles);

      if (compBuffManager.numGoals > 0 && sceneParams.showGoals){
        renderBuffManager.drawGoals(device, renderPass, compBuffManager.goalsBuffer, compBuffManager.numGoals);
      }

      renderPass.endPass();
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
