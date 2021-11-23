const scatterWidth = 100;
const diskRadius = 0.5;
const invMass = 0.0167;


export class ComputeBufferManager {
  numAgents : number;

  // buffer sizes
  simulationUBOBufferSize : number;

  // buffer item sizes (for buffers that might change size)
  agentInstanceSize : number;
  agentPositionOffset : number;
  agentColorOffset : number;

  device : GPUDevice;

  // buffers
  simulationUBOBuffer : GPUBuffer;
  agentsBuffer : GPUBuffer; // data on each agent, including position, velocity, etc.

  // bind group layout
  bindGroupLayout : GPUBindGroupLayout;

  constructor(device: GPUDevice, 
              numAgents: number){
    this.device = device;
    this.agentInstanceSize = 
    3 * 4 + // position
    1 * 4 + // radius
    4 * 4 + // color
    3 * 4 + // velocity
    1 * 4 + // inverse mass
    3 * 4 + // planned position
    1 * 4 + // padding
    3 * 4 + // goal
    1 * 4 + // padding
    20 * 4 + // neighbors, max 20
    0;

    this.agentPositionOffset = 0;
    this.agentColorOffset = 4 * 4;


    this.numAgents = numAgents;


    // --- set buffer sizes ---

    this.simulationUBOBufferSize =
      1 * 4 + // deltaTime
      3 * 4 + // padding
      4 * 4 + // seed
      0;

    this.initBuffers();

    this.setBindGroupLayout();
  }

  initBuffers(){
    // simulation parameter buffer
    this.simulationUBOBuffer = this.device.createBuffer({
      size: this.simulationUBOBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // agent buffer
    let initialAgentData = this.getAgentData(this.numAgents);
    this.agentsBuffer = this.device.createBuffer({
      size: this.numAgents * this.agentInstanceSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Float32Array(this.agentsBuffer.getMappedRange()).set(
      initialAgentData
    );
    this.agentsBuffer.unmap(); 
  }

  setGoal(){

    // a buffer of a single vec3 is padded into a vec4
    let goalDataSize = 4;
    const initialGoalData = new Float32Array(this.numAgents * goalDataSize);  

    let direction = -1;
    for (let i = 0; i < this.numAgents; ++i) {
          initialGoalData[i * goalDataSize + 0] = Math.random() * 0.5 * 2 - 1;
          initialGoalData[i * goalDataSize + 1] = 0.0;
          initialGoalData[i * goalDataSize + 2] = Math.random() * direction;
          if (i > this.numAgents/2){
            direction = 1
          }
    }
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

  setBindGroupLayout(){
  // create bindgroup layout
  this.bindGroupLayout = this.device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // simulationUBOBuffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform"
        }
      },
      {
        binding: 1, // agentsBuffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
    ]
  });
  }

  getBindGroup(){
    var computeBindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
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
            size: this.numAgents * this.agentInstanceSize,
          },
        },
      ],
    });
    return computeBindGroup;
  }

  getAgentData(numAgents: number){
    const agentIdxOffset = this.agentInstanceSize / 4;
    //48 is total byte size of each agent
    const initialAgentData = new Float32Array(numAgents * agentIdxOffset);  

    for (let i = 0; i < numAgents/2; ++i) {
      // position.xyz
      initialAgentData[agentIdxOffset * i + 0] = scatterWidth * (Math.random() - 0.5);
      initialAgentData[agentIdxOffset * i + 1] = 0.5;
      initialAgentData[agentIdxOffset * i + 2] = scatterWidth * 0.5 + 2 * (Math.random() - 0.5);

      // radius
      initialAgentData[agentIdxOffset * i + 3] = diskRadius;

      // color.rgba
      initialAgentData[agentIdxOffset * i + 4] = 1;
      initialAgentData[agentIdxOffset * i + 5] = 0;
      initialAgentData[agentIdxOffset * i + 6] = 0;
      initialAgentData[agentIdxOffset * i + 7] = 1;

      // velocity.xyz
      initialAgentData[agentIdxOffset * i + 8] = 0;
      initialAgentData[agentIdxOffset * i + 9] = 0;
      initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random())*-1;

      // inverse mass
      initialAgentData[agentIdxOffset * i + 11] = invMass;

      // goal
      initialAgentData[agentIdxOffset * i + 16] = 0;
      initialAgentData[agentIdxOffset * i + 17] = 0;
      initialAgentData[agentIdxOffset * i + 18] = -scatterWidth;
    }
    for (let i = numAgents/2; i < numAgents; ++i) {
      // position.xyz
      initialAgentData[agentIdxOffset * i + 0] = -scatterWidth * (Math.random() - 0.5);
      initialAgentData[agentIdxOffset * i + 1] = 0.5;
      initialAgentData[agentIdxOffset * i + 2] = -scatterWidth * 0.5 + 2 * (Math.random() - 0.5);

      // radius
      initialAgentData[agentIdxOffset * i + 3] = diskRadius;
      
      // color.rgba
      initialAgentData[agentIdxOffset * i + 4] = 0;
      initialAgentData[agentIdxOffset * i + 5] = 0;
      initialAgentData[agentIdxOffset * i + 6] = 1;
      initialAgentData[agentIdxOffset * i + 7] = 1;

      // velocity.xyz
      initialAgentData[agentIdxOffset * i + 8] = 0;
      initialAgentData[agentIdxOffset * i + 9] = 0;
      initialAgentData[agentIdxOffset * i + 10] = (0.1 + 0.5 * Math.random());

      // inverse mass
      initialAgentData[agentIdxOffset * i + 11] = invMass;

      initialAgentData[agentIdxOffset * i + 16] = 0;
      initialAgentData[agentIdxOffset * i + 17] = 0;
      initialAgentData[agentIdxOffset * i + 18] = scatterWidth;
    }
    return initialAgentData;
  }

async readAgent(commandEncoder, i : number){
    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = this.device.createBuffer({
      size: this.agentInstanceSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      this.agentsBuffer, // source buffer
      0, // source offset
      gpuReadBuffer, // destination buffer
      0, // destination offset
      this.agentInstanceSize // size
    );
  
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
  
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    console.log(new Float32Array(arrayBuffer));
}
}
