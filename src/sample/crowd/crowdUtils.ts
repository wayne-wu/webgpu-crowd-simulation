const scatterWidth = 100;

export class ComputeBufferManager {
  numAgents : number;

  // buffer sizes
  simulationUBOBufferSize : number;

  // buffer item sizes (for buffers that might change size)
  agentInstanceSize : number;
  goalBufferItemSize : number;
  gridCellBufferItemSize : number;

  device : GPUDevice;

  // buffers
  simulationUBOBuffer : GPUBuffer;
  agentsBuffer : GPUBuffer;
  gridCellBuffer : GPUBuffer;
  goalBuffer : GPUBuffer; // currently the preferred velocity, could be location sought

  // bind group layout
  bindGroupLayout : GPUBindGroupLayout;

  constructor(device: GPUDevice, 
              numAgents: number, 
              agentInstanceSize : number){
    this.device = device;
    this.agentInstanceSize = agentInstanceSize;
    this.numAgents = numAgents;


    // --- set buffer sizes ---

    this.simulationUBOBufferSize =
      1 * 4 + // deltaTime
      3 * 4 + // padding
      4 * 4 + // seed
      0;

    // --- set item sizes (for buffers that might change size) ---
    this.goalBufferItemSize = (3 + 1) * 4; // a vec3<f32> plus 1 byte of padding per agent
    this.gridCellBufferItemSize = 4; // a u32 per agent


    this.initBuffers();

    this.setBindGroupLayout();
  }

  initBuffers(){
    // simulation parameter buffer
    this.simulationUBOBuffer = this.device.createBuffer({
      size: this.simulationUBOBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // agent goal buffer
    this.goalBuffer = this.device.createBuffer({
      size: this.numAgents * this.goalBufferItemSize,
      // use STORAGE because it's not a uniform
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      // note we're mapped at creation so we can initialize
      // data on the CPU side
      mappedAtCreation: true
    });
    this.setGoal();
    this.goalBuffer.unmap(); // send data/control over to GPU

    // grid cell buffer
    this.gridCellBuffer = this.device.createBuffer({
      size: this.numAgents * this.gridCellBufferItemSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

    new Float32Array(this.goalBuffer.getMappedRange()).set(initialGoalData);
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
      {
        binding: 2, // goal buffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
      {
        binding: 3, // grid cell buffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      }
    ]
  });
  }

  getBindGroup(computePipeline){
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
        {
          binding: 2,
          resource: {
            buffer: this.goalBuffer,
            offset: 0,
            size: this.numAgents * this.goalBufferItemSize,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.gridCellBuffer,
            offset: 0,
            size: this.numAgents * this.gridCellBufferItemSize,
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
