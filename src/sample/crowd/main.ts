import { mat4, vec3, vec4 } from 'gl-matrix';
import { makeSample, SampleInit } from '../../components/SampleLayout';
import Camera from "./Camera";

import { TestScene, ComputeBufferManager } from './crowdUtils';
import { RenderBufferManager } from './renderUtils';

import renderWGSL from '../../shaders/background.render.wgsl';
import crowdWGSL from '../../shaders/crowd.render.wgsl';
import explicitIntegrationWGSL from '../../shaders/explicitIntegration.compute.wgsl';
import bitonicSortWGSL from '../../shaders/bitonicSort.compute.wgsl';
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

const gpuProfilerPasses = [
  'compute',
  'shadow',
  'render',
] as const;

type GpuProfilerPassName = typeof gpuProfilerPasses[number];

type GpuProfiler = {
  supported: boolean,
  active: boolean,
  querySet: GPUQuerySet | null,
  resolveBuffer: GPUBuffer | null,
  readbackBuffer: GPUBuffer | null,
  sampleCount: number,
  frameInterval: number,
  framesUntilReadback: number,
  readbackPending: boolean,
  capturePending: boolean,
  totalsMs: Record<GpuProfilerPassName, number>,
  samplesAccumulated: number,
};

function createGpuProfiler(device: GPUDevice, enableTimestamps: boolean): GpuProfiler {
  if (!enableTimestamps) {
    return {
      supported: false,
      active: false,
      querySet: null,
      resolveBuffer: null,
      readbackBuffer: null,
      sampleCount: 0,
      frameInterval: 30,
      framesUntilReadback: 30,
      readbackPending: false,
      capturePending: false,
      totalsMs: { compute: 0, shadow: 0, render: 0 },
      samplesAccumulated: 0,
    };
  }

  const sampleCount = gpuProfilerPasses.length * 2;
  const bufferSize = sampleCount * BigUint64Array.BYTES_PER_ELEMENT;

  return {
    supported: true,
    active: false,
    querySet: device.createQuerySet({
      type: 'timestamp',
      count: sampleCount,
    }),
    resolveBuffer: device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readbackBuffer: device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    sampleCount,
    frameInterval: 30,
    framesUntilReadback: 30,
    readbackPending: false,
    capturePending: false,
    totalsMs: { compute: 0, shadow: 0, render: 0 },
    samplesAccumulated: 0,
  };
}

function getTimestampWrites(profiler: GpuProfiler, passIndex: number) {
  if (!profiler.active || profiler.querySet == null) {
    return undefined;
  }

  return {
    querySet: profiler.querySet,
    beginningOfPassWriteIndex: passIndex * 2,
    endOfPassWriteIndex: passIndex * 2 + 1,
  };
}

function scheduleGpuProfilerReadback(device: GPUDevice, profiler: GpuProfiler) {
  if (!profiler.active || profiler.readbackPending || profiler.readbackBuffer == null) {
    return;
  }
  profiler.readbackPending = true;
  profiler.capturePending = false;

  device.queue.onSubmittedWorkDone().then(async () => {
    if (profiler.readbackBuffer == null) {
      profiler.readbackPending = false;
      return;
    }

    try {
      await profiler.readbackBuffer.mapAsync(GPUMapMode.READ);
      const mapped = profiler.readbackBuffer.getMappedRange();
      const timestamps = new BigUint64Array(mapped.slice(0));
      profiler.readbackBuffer.unmap();

      for (let i = 0; i < gpuProfilerPasses.length; i++) {
        const passName = gpuProfilerPasses[i];
        const begin = timestamps[i * 2];
        const end = timestamps[i * 2 + 1];
        profiler.totalsMs[passName] += Number(end - begin) / 1e6;
      }

      profiler.samplesAccumulated++;
      if (profiler.samplesAccumulated >= 10) {
        console.log('[gpu profiler]', {
          computeMs: profiler.totalsMs.compute / profiler.samplesAccumulated,
          shadowMs: profiler.totalsMs.shadow / profiler.samplesAccumulated,
          renderMs: profiler.totalsMs.render / profiler.samplesAccumulated,
        });
        profiler.totalsMs = { compute: 0, shadow: 0, render: 0 };
        profiler.samplesAccumulated = 0;
      }
    } catch (error) {
      console.warn('GPU profiler readback failed.', error);
      profiler.active = false;
    } finally {
      profiler.readbackPending = false;
    }
  });
}

function updateGpuProfilerCaptureState(profiler: GpuProfiler) {
  if (!profiler.active || profiler.readbackPending || profiler.capturePending) {
    return false;
  }

  profiler.framesUntilReadback--;
  if (profiler.framesUntilReadback > 0) {
    return false;
  }

  profiler.framesUntilReadback = profiler.frameInterval;
  profiler.capturePending = true;
  return true;
}

function setGpuProfilerActive(profiler: GpuProfiler, active: boolean) {
  profiler.active = profiler.supported && active;
  profiler.framesUntilReadback = profiler.frameInterval;
  profiler.capturePending = false;
  profiler.totalsMs = { compute: 0, shadow: 0, render: 0 };
  profiler.samplesAccumulated = 0;
}

// Reset camera to original settings (gui function)
function resetCameraFunc(x: number = 50, y: number = 50, z: number = 50) {
  camera = new Camera(vec3.fromValues(x, y, z), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

function fillSortPipelineList(device,
                              numAgents : number, 
                              computePipelinesSort,
                              compBuffManager){

    // be sure the list is empty before pushing new pipelines
    computePipelinesSort.length = 0;

    var pipelineLayout =  device.createPipelineLayout({
      bindGroupLayouts: [compBuffManager.bindGroupLayout]
    });
    var shaderModule = device.createShaderModule({
      code: headerWGSL + bitonicSortWGSL,
    });

    // set up sort pipelines
    // adapted from Wikipedia's non-recursive example of bitonic sort:
    // https://en.wikipedia.org/wiki/Bitonic_sorter
    for (let k = 2; k <= numAgents; k <<= 1){ // k is doubled every iteration
      for (let j = k >> 1; j > 0; j >>= 1){ // j is halved at every iteration, with truncation of fractional parts
        computePipelinesSort.push(
          device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
              module: shaderModule,
              entryPoint: 'main',
              constants: {
                1100: j,
                1200: k,
              }
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
    model: 'Duck',
    showGoals: true,
    shadowOn: true,
    debugCell: false,
    gpuProfiling: false,
    'total agents': "", // dummy, autofilled later
    '2^x agents': 10
  }

  const simulationParams = {
    simulate: true,
    step: () => {
      if (!simulationParams.simulate) {
        pendingSingleStep = true;
      }
    },
    deltaTime: 0.02,
    stabilityIterations: 1,
    constraintIterations: 6,
    numObstacles : 1,
    avoidanceModel: false,
    lookAhead : 6.0,
    gridWidth: guiParams.gridWidth,
    resetSimulation: resetSim,
  };

  // GUI GLOBALS ------------------------------------------------------------
  // default don't display slider to select number of agents -- will re-add if scene requires
  let numAgentsSliderDisplayed = false;
  let numAgentsController = null;
  let pendingSingleStep = false;
  let lastConfiguredScene = sceneParams.scene;


  // GUI ELEMENTS -----------------------------------------------------------
  const gridFolder = gui.addFolder("Grid");
  gridFolder.add(guiParams, 'gridWidth', 1, 5000, 1).onFinishChange(resetSim);
  gridFolder.add(guiParams, 'gridOn');
  gridFolder.open();

  const camFolder = gui.addFolder("Camera");
  camFolder.add(guiParams, 'resetCamera');
  camFolder.open();

  const models = Array.from(Object.keys(meshDictionary));
  models.push('Cube');

  const sceneFolder = gui.addFolder("Scene");
  sceneFolder.add(sceneParams, 'scene', Object.values(TestScene)).onFinishChange(resetSim);
  sceneFolder.add(sceneParams, 'model', models).onFinishChange(resetRender);
  sceneFolder.add(sceneParams, 'showGoals');
  sceneFolder.add(sceneParams, 'shadowOn');
  sceneFolder.add(sceneParams, 'total agents');
  sceneFolder.open();
  
  const simFolder = gui.addFolder("Simulation");
  const simulateController = simFolder.add(simulationParams, 'simulate');
  const stepController = simFolder.add(simulationParams, 'step');
  simFolder.add(simulationParams, 'resetSimulation');
  simFolder.add(simulationParams, 'deltaTime', 0.0001, 1.0, 0.0001);
  simFolder.add(simulationParams, 'stabilityIterations', 1, 10, 1).onFinishChange(resetSim);
  simFolder.add(simulationParams, 'constraintIterations', 1, 10, 1).onFinishChange(resetSim);
  simFolder.add(simulationParams, 'lookAhead', 3.0, 15.0, 1.0);
  simFolder.add(simulationParams, 'avoidanceModel');

  simFolder.open();

  function updateStepControllerState() {
    const disabled = simulationParams.simulate;
    const style = stepController.domElement.style;
    style.pointerEvents = disabled ? 'none' : 'auto';
    style.opacity = disabled ? '0.5' : '1.0';
  }

  simulateController.onChange(() => {
    updateStepControllerState();
  });
  updateStepControllerState();

  const debugFolder = gui.addFolder("Debug");
  debugFolder.add(sceneParams, 'debugCell');
  debugFolder.open();

  // manually set text for total number of agents 
  const totalAgentsDOM = gui.__folders["Scene"].__controllers[4].domElement;
  totalAgentsDOM.innerHTML = totalAgentsDOM.innerHTML.substr(0, totalAgentsDOM.innerHTML.length - 1) + "disabled=\"true\">";
  totalAgentsDOM.innerText = "1024";


  /////////////////////////////////////////////////////////////////////////
  //                     Initial Context Setup                           //
  /////////////////////////////////////////////////////////////////////////

  const adapter = await navigator.gpu.requestAdapter();
  if (adapter == null) {
    throw new Error('Failed to acquire a WebGPU adapter.');
  }

  const requiredFeatures : GPUFeatureName[] = [];
  const adapterFeatures = adapter.features as unknown as Set<GPUFeatureName>;
  const hasTimestampQuery = adapterFeatures.has('timestamp-query');
  if (hasTimestampQuery) {
    requiredFeatures.push('timestamp-query');
  } else {
    console.info('GPU timestamp queries are unavailable on this adapter/browser.');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
  });
  const gpuProfiler = createGpuProfiler(device, hasTimestampQuery);
  if (!gpuProfiler.supported) {
    sceneParams.gpuProfiling = false;
  }

  const gpuProfilingController = debugFolder.add(sceneParams, 'gpuProfiling');
  gpuProfilingController.name('gpuProfiling');
  if (!gpuProfiler.supported) {
    const style = gpuProfilingController.domElement.style;
    style.pointerEvents = 'none';
    style.opacity = '0.5';
  }
  gpuProfilingController.onChange((enabled: boolean) => {
    setGpuProfilerActive(gpuProfiler, enabled);
  });

  if (canvasRef.current === null) return;
  const context = canvasRef.current.getContext('webgpu');
  
  const devicePixelRatio = window.devicePixelRatio || 1;
  const presentationSize = [
    canvasRef.current.clientWidth * devicePixelRatio,
    canvasRef.current.clientHeight * devicePixelRatio,
  ];

  canvasRef.current.width = presentationSize[0];
  canvasRef.current.height = presentationSize[1];

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque',
  });

  camera = new Camera(vec3.fromValues(50, 50, 50), vec3.fromValues(0, 0, 0));
  aspect = canvasRef.current.width / canvasRef.current.height;
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();


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

  //////////////////////////////////////////////////////////////////////////////
  // Create Compute Pipelines
  //////////////////////////////////////////////////////////////////////////////
  const computeShadersPreSort = [
    headerWGSL + explicitIntegrationWGSL, 
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

  var computeWorkgroupCount = 0;
  var sortWorkgroupCount = 0;

  function createComputePipelines() {

    computeWorkgroupCount = Math.ceil(compBuffManager.numAgents/64);
    sortWorkgroupCount = Math.ceil(compBuffManager.numAgents/256);

    computePipelinesPreSort.splice(0);
    computePipelinesSort.splice(0);
    computePipelinesPostSort.splice(0);

    var pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [compBuffManager.bindGroupLayout]
    });

    // set up pre-sort pipelines
    for(let i = 0; i < computeShadersPreSort.length; i++){
      computePipelinesPreSort.push( 
          device.createComputePipeline({
          layout: pipelineLayout,
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
    let i = 0;
    for(;i < 2;){
      computePipelinesPostSort.push( 
          device.createComputePipeline({
          layout: pipelineLayout,
          compute: {
            module: device.createShaderModule({
              code: computeShadersPostSort[i++],
            }),
            entryPoint: 'main',
          },
        })
      );
    }

    var shaderModule = device.createShaderModule({
      code: computeShadersPostSort[i++],
    });
    for(let j = 0; j < simulationParams.constraintIterations; j++){
      computePipelinesPostSort.push( 
        device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
          constants: {
            1000 : j + 1,
          }
        },
      })
      );
    }

    for(;i < computeShadersPostSort.length;){
      computePipelinesPostSort.push( 
        device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: device.createShaderModule({
            code: computeShadersPostSort[i++],
          }),
          entryPoint: 'main',
        },
      })
      );
    }
  }

  function resetRender(){

    renderBuffManager = null;
    if (sceneParams.model == 'Cube'){
      const mesh = new Mesh(cubeVertexArray);
      mesh.scale = 0.2;
      renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
        presentationFormat, presentationSize,
        compBuffManager, mesh, gridTexture, sampler, 
        sceneParams.showGoals);
    } else {
    var modelData = meshDictionary[sceneParams.model];
    loadModel(modelData.filename, device).then((mesh : Mesh) => {
      mesh.scale = modelData.scale;
      renderBuffManager = new RenderBufferManager(device, guiParams.gridWidth, 
        presentationFormat, presentationSize,
        compBuffManager, mesh, gridTexture, sampler, 
        sceneParams.showGoals);
      });
    }
  }

  function resetSim(){
    const sceneChanged = sceneParams.scene !== lastConfiguredScene;

    switch(sceneParams.scene) {
      case TestScene.PROXIMAL:
        setTestScene(vec3.fromValues(5, 10, 5), false, 6, 30, 0, true, sceneChanged);
        break;
      case TestScene.BOTTLENECK:
        setTestScene(vec3.fromValues(20, 20, 20), false, 9, 63, 2, true, sceneChanged);
        break;
      case TestScene.DENSE:
        setTestScene(vec3.fromValues(80, 75, 0), true, 15, 1000, 0, false, sceneChanged);
        break;
      case TestScene.SPARSE:
        setTestScene(vec3.fromValues(50, 50, 50), true, 12, 100, 0, false, sceneChanged);
        break;
      case TestScene.OBSTACLES:
        setTestScene(vec3.fromValues(50, 50, 50), false, 10, 50, 5, true, sceneChanged);
        break;
      case TestScene.CIRCLE:
        setTestScene(vec3.fromValues(5, 20, 5), false, 6, 20, 0, true, sceneChanged);
        break;
      case TestScene.DISPERSED:
        setTestScene(vec3.fromValues(0, 60, 0), false, 11, 100, 0, false, sceneChanged);
        break;
    }

    compBuffManager.testScene = sceneParams.scene;
    compBuffManager.numValidAgents = 1<<sceneParams['2^x agents'];
    totalAgentsDOM.innerText = compBuffManager.numValidAgents + "";

    compBuffManager.gridWidth = simulationParams.gridWidth;

    // NOTE: Can't have 0 binding size so we just set to 1 dummy if no obstacles
    compBuffManager.numObstacles = Math.max(simulationParams.numObstacles, 1);

    // reinitilize buffers based on the new number of agents
    compBuffManager.initBuffers();

    computeBindGroup1 = compBuffManager.getBindGroup(false, "R1W2");  // READ agents1 WRITE agents2
    computeBindGroup2 = compBuffManager.getBindGroup(true, "R2W1");   // READ agents2 WRITE agents1
    computeBindGroup = computeBindGroup1;

    // the number of steps in the sort pipeline is proportional
    // to log2 the number of agents, so reinitiliaze it
    createComputePipelines();
    lastConfiguredScene = sceneParams.scene;
  }

  function setTestScene(camPos: vec3, displayAgentSlider: boolean, numAgents: number, 
                        scenePlatformWidth: number, numObstacles: number, shadowOn: boolean, sceneChanged: boolean){
    resetCameraFunc(camPos[0], camPos[1], camPos[2]); // set scene's camera position
    if (!displayAgentSlider || sceneChanged) {
      sceneParams['2^x agents'] = numAgents;  // number of agents displayed in GUI
    }
    compBuffManager.numValidAgents = 1<<sceneParams['2^x agents'];    // number of agents to use in simulation
    platformWidth = scenePlatformWidth;               // size of the platform
    simulationParams.numObstacles = numObstacles;     // number of obstacles (used in compBufferManager, not gui)
    sceneParams.shadowOn = shadowOn;                      // display shadows on chosen scene

    // if agent slider exists and this scene doesn't support it, remove
    if (!displayAgentSlider && numAgentsSliderDisplayed) {
      sceneFolder.remove(numAgentsController);
      numAgentsController = null;
      numAgentsSliderDisplayed = false;
    }
    // if agent slider is supported and it doesn't exist, add it
    else if (displayAgentSlider && !numAgentsSliderDisplayed) {
      numAgentsController = sceneFolder
        .add(sceneParams, '2^x agents', 1, 20, 1)
        .listen()
        .onFinishChange(resetSim);
      numAgentsSliderDisplayed = true;
    }
    
    // set camera reset function used in GUI (reset to this scene's default camera position)
    guiParams.resetCamera = () => resetCameraFunc(camPos[0], camPos[1], camPos[2]);
  }

  // get compute bind group
  var computeBindGroup1 = compBuffManager.getBindGroup(false, "R1W2");
  var computeBindGroup2 = compBuffManager.getBindGroup(true, "R2W1");

  var computeBindGroup = computeBindGroup1;

  function pingPongBuffer(){
    if (computeBindGroup == computeBindGroup1)
      computeBindGroup = computeBindGroup2;
    else if (computeBindGroup == computeBindGroup2)
      computeBindGroup = computeBindGroup1;
  }

  resetSim();
  resetRender();

  var time = 0;
  function frame() {

    time++;
    stats.begin();

    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    camera.update();
    const captureGpuProfile = updateGpuProfilerCaptureState(gpuProfiler);

    //------------------ Compute Calls ------------------------ //
    {
      var command = device.createCommandEncoder();

      const shouldRunSimulation = simulationParams.simulate || pendingSingleStep;
      if(shouldRunSimulation) {

        // write the parameters to the Uniform buffer for our compute shaders
        compBuffManager.writeSimParams(simulationParams, true);

        // execute each compute shader in the order they were pushed onto
        // the computePipelines array
        var passEncoder = command.beginComputePass({
          timestampWrites: getTimestampWrites(gpuProfiler, 0),
        } as unknown as GPUComputePassDescriptor);
        passEncoder.setBindGroup(0, computeBindGroup);

        //// ----- Compute Pass Before Sort -----
        for (let i = 0; i < computePipelinesPreSort.length; i++){
          passEncoder.setPipeline(computePipelinesPreSort[i]);
          passEncoder.dispatchWorkgroups(computeWorkgroupCount);
        }

        // ----- Compute Pass Sort -----
        for (let i = 0; i < computePipelinesSort.length; i++){
          passEncoder.setPipeline(computePipelinesSort[i]);
          passEncoder.dispatchWorkgroups(sortWorkgroupCount);
        }
        
        // ----- Compute Pass Post Sort 1 -----
        let i = 0;
        for (;i < 1 /* constraint shader index */; i++){
          passEncoder.setPipeline(computePipelinesPostSort[i]);
          passEncoder.dispatchWorkgroups(computeWorkgroupCount);
        }     
        
        // Stability Iterations
        passEncoder.setPipeline(computePipelinesPostSort[i++]);
        for (let j = 0; j < simulationParams.stabilityIterations; j++){
          passEncoder.setBindGroup(0, computeBindGroup);
          passEncoder.dispatchWorkgroups(computeWorkgroupCount);

          pingPongBuffer();
        }
        
        // ----- Compute Pass Constraint Solve -----
        for (let j = 0; j < simulationParams.constraintIterations && i < computePipelinesPostSort.length; j++) {
          passEncoder.setPipeline(computePipelinesPostSort[i++]);
          passEncoder.setBindGroup(0, computeBindGroup);
          passEncoder.dispatchWorkgroups(computeWorkgroupCount);

          pingPongBuffer();
        }      

        passEncoder.setBindGroup(0, computeBindGroup);

        // ----- Compute Pass Post Sort 2 -----
        for (;i < computePipelinesPostSort.length; i++){
          passEncoder.setPipeline(computePipelinesPostSort[i]);
          passEncoder.dispatchWorkgroups(computeWorkgroupCount);
        }

        pingPongBuffer();

        passEncoder.end();

        pendingSingleStep = false;
      }
    }

    // ------------------ Render Calls ------------------------- //
    if (renderBuffManager != null) {

      renderBuffManager.updateSceneUBO(camera, guiParams.gridOn, time, sceneParams.shadowOn, sceneParams.debugCell);
      
      const agentsBuffer : GPUBuffer = computeBindGroup == computeBindGroup2 ? compBuffManager.agents2Buffer : compBuffManager.agents1Buffer;

      if(sceneParams.shadowOn) {
        const shadowPassDescriptor = {
          ...renderBuffManager.shadowPassDescriptor,
          timestampWrites: getTimestampWrites(gpuProfiler, 1),
        } as unknown as GPURenderPassDescriptor;
        renderBuffManager.drawCrowdShadow(device, command, agentsBuffer, compBuffManager.numAgents, shadowPassDescriptor);
      }

      // const transformationMatrix = getTransformationMatrix();
      renderBuffManager.renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();
      
      const renderPassDescriptor = {
        ...renderBuffManager.renderPassDescriptor,
        timestampWrites: getTimestampWrites(gpuProfiler, 2),
      } as unknown as GPURenderPassDescriptor;
      const renderPass = command.beginRenderPass(renderPassDescriptor);

      // ----------------------- Draw ------------------------- //
      renderBuffManager.drawPlatform(device, renderPass, platformWidth);
      
      renderBuffManager.drawCrowd(device, renderPass, agentsBuffer, compBuffManager.numAgents);

      if (simulationParams.numObstacles > 0)
        renderBuffManager.drawObstacles(device, renderPass, compBuffManager.obstaclesBuffer, compBuffManager.numObstacles);

      if (compBuffManager.numGoals > 0 && sceneParams.showGoals){
        renderBuffManager.drawGoals(device, renderPass, compBuffManager.goalsBuffer, compBuffManager.numGoals);
      }

      renderPass.end();
    }

    if (captureGpuProfile && gpuProfiler.querySet != null && gpuProfiler.resolveBuffer != null && gpuProfiler.readbackBuffer != null) {
      const bytesPerTimestamp = BigUint64Array.BYTES_PER_ELEMENT;
      command.resolveQuerySet(gpuProfiler.querySet, 0, gpuProfiler.sampleCount, gpuProfiler.resolveBuffer, 0);
      command.copyBufferToBuffer(
        gpuProfiler.resolveBuffer,
        0,
        gpuProfiler.readbackBuffer,
        0,
        gpuProfiler.sampleCount * bytesPerTimestamp
      );
    }

    device.queue.submit([command.finish()]);
    if (captureGpuProfile) {
      scheduleGpuProfilerReadback(device, gpuProfiler);
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
