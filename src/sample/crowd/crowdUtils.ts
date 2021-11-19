const scatterWidth = 100;

export class ComputeBufferManager {
  numAgents : number;
  agentInstanceByteSize : number;
  simulationUBOBufferSize : number;

  device : GPUDevice;

  // buffers
  simulationUBOBuffer : GPUBuffer;
  agentsBuffer : GPUBuffer;
  gridCellBuffer : GPUBuffer;

  constructor(device: GPUDevice, 
              numAgents: number, 
              agentInstanceByteSize : number){
    this.device = device;
    this.agentInstanceByteSize = agentInstanceByteSize;
    this.numAgents = numAgents;

    this.simulationUBOBufferSize =
      1 * 4 + // deltaTime
      3 * 4 + // padding
      4 * 4 + // seed
      0;

      this.initBuffers();
  }

  initBuffers(){
    this.simulationUBOBuffer = this.device.createBuffer({
      size: this.simulationUBOBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.gridCellBuffer = this.device.createBuffer({
      size: this.simulationUBOBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    let initialAgentData = this.getAgentData(this.numAgents);
    this.agentsBuffer = this.device.createBuffer({
      size: this.numAgents * this.agentInstanceByteSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Float32Array(this.agentsBuffer.getMappedRange()).set(
      initialAgentData
    );
    this.agentsBuffer.unmap();
  }

  writeSimParams(simulationParams){

      this.device.queue.writeBuffer(
        this.simulationUBOBuffer,
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
  }

  getBindGroup(computePipeline){
    var computeBindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.simulationUBOBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.agentsBuffer,
            offset: 0,
            size: this.numAgents * this.agentInstanceByteSize,
          },
        },
      ],
    });
    return computeBindGroup;
  }

  getAgentData(numAgents: number){
    const agentIdxOffset = 12;
    //48 is total byte size of each agent
    const initialAgentData = new Float32Array(numAgents * agentIdxOffset);  

    for (let i = 0; i < numAgents/2; ++i) {
      // position.xyz
      initialAgentData[agentIdxOffset * i + 0] = scatterWidth * (Math.random() - 0.5);
      initialAgentData[agentIdxOffset * i + 1] = 0.5;
      initialAgentData[agentIdxOffset * i + 2] = scatterWidth + 2 * (Math.random() - 0.5);

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
      initialAgentData[agentIdxOffset * i + 0] = -scatterWidth * (Math.random() - 0.5);
      initialAgentData[agentIdxOffset * i + 1] = 0.5;
      initialAgentData[agentIdxOffset * i + 2] = -scatterWidth + 2 * (Math.random() - 0.5);

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
    return initialAgentData;
  }
}