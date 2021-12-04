import { mat4, vec3 } from "gl-matrix";

const scatterWidth = 100;
const diskRadius = 0.5;
const invMass = 0.5;
const minY = 0.5;


export class ComputeBufferManager {
  numAgents : number;
  numObstacles : number;

  // buffer sizes
  simulationUBOBufferSize : number;

  // buffer item sizes (for buffers that might change size)
  agentInstanceSize : number;
  agentPositionOffset : number;
  agentColorOffset : number;
  cellInstanceSize : number;

  obstacleInstanceSize : number;
  obstaclePositionOffset : number;

  device : GPUDevice;

  // buffers
  simulationUBOBuffer : GPUBuffer;
  agentsBuffer : GPUBuffer;           // data on each agent, including position, velocity, etc.
  cellsBuffer : GPUBuffer;            // start / end indices for each cell (in pairs)
  
  obstaclesBuffer : GPUBuffer;

  // bind group layout
  bindGroupLayout : GPUBindGroupLayout;

  gridWidth : number;
  gridHeight : number;

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
    1 * 4 + // cell
    20 * 4 + // close neighbors
    20 * 4 + // far neighbors
    0;

    this.agentPositionOffset = 0;
    this.agentColorOffset = 4 * 4;

    this.numAgents = numAgents;
    
    this.numObstacles = 1;
    this.obstacleInstanceSize =
      3 * 4 + // position
      1 * 4 + // rotation-y
      3 * 4 + // scale
      1 * 4 + // padding
      0;

    this.simulationUBOBufferSize =
      1 * 4 + // deltaTime
      1 * 4 + // avoidance (int)
      2 * 4 + // padding
      4 * 4 + // seed
      0;

    this.cellInstanceSize = 
      2 * 4   // 2 u32 indices per pair of start/end ptrs
    ;

    this.gridWidth = 100;
    this.gridHeight = 50;

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
 
    // cells buffer
    this.cellsBuffer = this.device.createBuffer({
      size: this.gridWidth * this.gridHeight * this.cellInstanceSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });

    let obstacleData = this.initObstacles(this.numObstacles);
    this.obstaclesBuffer = this.device.createBuffer({
      size: this.numObstacles * this.obstacleInstanceSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Float32Array(this.obstaclesBuffer.getMappedRange()).set(obstacleData);
    this.obstaclesBuffer.unmap();
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
        binding: 2, // obstacleBuffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      }
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
        {
          binding: 2,
          resource: {
            buffer: this.obstaclesBuffer,
            offset: 0,
            size: this.numObstacles * this.obstacleInstanceSize,
          },
        }
      ],
    });
    return computeBindGroup;
  }

  // Initialize obstacles
  initObstacles(numObstacles: number) {
    const obstacleData = new Float32Array(numObstacles * this.obstacleInstanceSize / 4);

    for(let i = 0; i < numObstacles; i++){
      let offset = this.obstacleInstanceSize * i / 4;

      obstacleData[offset + 0] = 0; //100*(Math.random()-0.5);
      obstacleData[offset + 1] = minY;
      obstacleData[offset + 2] = 0;

      obstacleData[offset + 3] = 0.785; //rotation

      obstacleData[offset + 4] = 3.0;
      obstacleData[offset + 5] = 1.0;
      obstacleData[offset + 6] = 3.0;
    }
    
    return obstacleData;
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

    agents[offset + 4] = Math.random(); //color[0];
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
      let x = i%10 - 5;
      let z = Math.floor(i/10) + 10;
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

}
