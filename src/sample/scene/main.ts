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
  getUniformBindGroup
} from './renderUtils';

import renderWGSL from './shaders.wgsl';
import { getuid } from 'process';

let camera : Camera;
let aspect : number;

function resetCameraFunc() {
  camera = new Camera(vec3.fromValues(3, 3, 3), vec3.fromValues(0, 0, 0));
  camera.setAspectRatio(aspect);
  camera.updateProjectionMatrix();
}

const init: SampleInit = async ({ canvasRef, gui }) => {

  // create camera
  camera = new Camera(vec3.fromValues(3, 3, 3), vec3.fromValues(0, 0, 0));

  const guiParams = {
    gridWidth: 5,
    resetCamera: resetCameraFunc
  };

  let prevGridWidth = 5;
  gui.addFolder("Grid");
  gui.add(guiParams, 'gridWidth', 1, 100, 1);
  gui.open();
  gui.addFolder("Camera");
  gui.add(guiParams, 'resetCamera');
  gui.open();

  aspect = canvasRef.current.width / canvasRef.current.height;
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

  // Create vertex buffers for the platform and the grid lines
  const verticesBufferPlatform = getVerticesBuffer(device, platformVertexArray);
  // Compute the grid lines based on an input gridWidth
  let gridLinesVertexArray = getGridLines(guiParams.gridWidth);
  let verticesBufferGridLines = getVerticesBuffer(device, gridLinesVertexArray);

  // Create render pipelines for platform and grid lines
  const pipelinePlatform = getPipeline(
        device, renderWGSL, 'vs_main', 'fs_platform', platformVertexSize,
        platformPositionOffset, platformUVOffset, presentationFormat, 'triangle-list', 'back'
  );
  const pipelineGridLines = getPipeline(
        device, renderWGSL, 'vs_main', 'fs_gridLines', gridLinesVertexSize,
        gridLinesPositionOffset, gridLinesUVOffset, presentationFormat, 'line-list', 'none'
  );

  // Get the depth texture for both pipelines
  const depthTexture = getDepthTexture(device, presentationSize);

  const uniformBufferPlatform = getUniformBuffer(device);
  const uniformBufferGridLines = getUniformBuffer(device);
  
  const uniformBindGroupPlatform = getUniformBindGroup(device, pipelinePlatform, uniformBufferPlatform);
  const uniformBindGroupGridLines = getUniformBindGroup(device, pipelineGridLines, uniformBufferGridLines);

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

  function getTransformationMatrix() {
    const modelMatrix = mat4.create();
    mat4.identity(modelMatrix);
    mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(50, 0.1, 50));

    //return modelViewProjectionMatrix as Float32Array;
    const modelViewProjectionMatrix = mat4.create();
    mat4.multiply(modelViewProjectionMatrix, camera.viewMatrix, modelMatrix);
    mat4.multiply(modelViewProjectionMatrix, camera.projectionMatrix, modelViewProjectionMatrix);
    return modelViewProjectionMatrix as Float32Array;
  }

  function frame() {
    // Sample is no longer the active page.
    if (!canvasRef.current) return;

    if (prevGridWidth != guiParams.gridWidth) {
      gridLinesVertexArray = getGridLines(guiParams.gridWidth);
      verticesBufferGridLines = getVerticesBuffer(device, gridLinesVertexArray);
      prevGridWidth = guiParams.gridWidth;
    }

    camera.update();
    const transformationMatrix = getTransformationMatrix();
    device.queue.writeBuffer(
      uniformBufferPlatform,
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
    passEncoder.setPipeline(pipelinePlatform);
    passEncoder.setBindGroup(0, uniformBindGroupPlatform);
    passEncoder.setVertexBuffer(0, verticesBufferPlatform);
    passEncoder.draw(platformVertexCount, 1, 0, 0);
  
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