const scatterWidth = 100;
const diskRadius = 0.5;
const invMass = 0.5;
const minY = 0.5;


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
    20 * 4 + // close neighbors
    20 * 4 + // far neighbors
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
    //let initialAgentData = this.getAgentData(this.numAgents);
    let initialAgentData = this.initAgentsProximity(this.numAgents);
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
    //48 is total byte size of each agent
    const initialAgentData = new Float32Array(numAgents * this.agentInstanceSize / 4);  

    for (let i = 0; i < numAgents/2; ++i) {
      this.setAgentData(
        initialAgentData, i,
        [scatterWidth * (Math.random() - 0.5), scatterWidth * 0.25 + 2 * (Math.random() - 0.5)],
        [1,0,0,1], [0,(0.1 + 0.5 * Math.random())*-1], [0, -scatterWidth]);
    }

    for (let i = numAgents/2; i < numAgents; ++i) {
      this.setAgentData(
        initialAgentData, i,
        [-scatterWidth * (Math.random() - 0.5), -scatterWidth * 0.25 + 2 * (Math.random() - 0.5)],
        [0,0,1,1], [0,(0.1 + 0.5 * Math.random())], [0, scatterWidth]);
    }

    return initialAgentData;
  }

  setAgentData(
    agents : Float32Array, index : number, position : number[], color : number[], 
    velocity : number[], goal : number[]) {
    const offset = this.agentInstanceSize * index / 4;
    
    agents[offset + 0] = position[0];
    agents[offset + 1] = minY;  // Force pos.y to be 0.5.
    agents[offset + 2] = position[1];
     
    agents[offset + 3] = diskRadius;

    agents[offset + 4] = color[0];
    agents[offset + 5] = color[1];
    agents[offset + 6] = color[2];
    agents[offset + 7] = color[3];

    agents[offset + 8] = velocity[0];
    agents[offset + 9] = 0.0;  // Force vel-y to be 0.
    agents[offset + 10] = velocity[1];

    agents[offset + 11] = invMass;

    agents[offset + 16] = goal[0];
    agents[offset + 17] = minY;
    agents[offset + 18] = goal[1];
  }

  initAgentsProximity(numAgents : number) {
    const initialAgentData = new Float32Array(numAgents * this.agentInstanceSize / 4);
    for (let i = 0; i < numAgents/2; ++i) {
      let x = Math.floor(i/10);
      let z = Math.floor(10+i%10);
      let v = 0.5;
      this.setAgentData(
        initialAgentData, 2*i,
        [1.25+x, z], [1,0,0,1], [0,-v], [0, -scatterWidth]);
      this.setAgentData(
        initialAgentData, 2*i + 1,
        [-1.25+x, -z], [0,0,1,1], [0,v], [0, scatterWidth]);
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
