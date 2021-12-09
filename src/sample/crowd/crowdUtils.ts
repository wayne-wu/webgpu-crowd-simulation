import { mat4, vec3 } from "gl-matrix";

const scatterWidth = 100;
const diskRadius = 0.5;
const invMass = 0.5;
const minY = 0.5;
const obstacleHeight = 2.0;
const preferredVelocity = 1.4;


export enum TestScene {
  PROXIMAL = "proximal",
  BOTTLENECK = "bottleneck",
  DENSE = "dense",
  SPARSE = "sparse",
  OBSTACLES = "obstacles",
  CIRCLE = "circle",
}


export class ComputeBufferManager {
  testScene : TestScene;

  numAgents : number;
  numValidAgents : number; 
  numObstacles : number;

  // buffer sizes
  simulationUBOBufferSize : number;

  // buffer item sizes (for buffers that might change size)
  agentInstanceSize : number;
  agentPositionOffset : number;
  agentColorOffset : number;
  agentVelocityOffset: number;
  cellInstanceSize : number;

  obstacleInstanceSize : number;
  obstaclePositionOffset : number;

  device : GPUDevice;

  // buffers
  simulationUBOBuffer : GPUBuffer;
  agents1Buffer : GPUBuffer;           // data on each agent, including position, velocity, etc.
  agents2Buffer : GPUBuffer;
  cellsBuffer : GPUBuffer;            // start / end indices for each cell (in pairs)
  
  obstaclesBuffer : GPUBuffer;

  // bind group layout
  bindGroupLayout : GPUBindGroupLayout;

  gridWidth : number;

  constructor(device: GPUDevice,
              testScene: TestScene, 
              numAgents: number,
              gridWidth: number){
    this.device = device;
    this.testScene = testScene;

    this.agentPositionOffset = 0;
    this.agentColorOffset = 4 * 4;
    this.agentVelocityOffset = 20 * 4;

    this.numAgents = Math.pow(2, Math.ceil(Math.log2(numAgents)));
    this.numValidAgents = numAgents;
    
    this.numObstacles = 1;

    // --- set buffer sizes ---

    this.agentInstanceSize = 
      3 * 4 + // position
      1 * 4 + // radius
      4 * 4 + // color
      3 * 4 + // velocity
      1 * 4 + // inverse mass
      3 * 4 + // planned position
      1 * 4 + // preferred speed
      3 * 4 + // goal
      1 * 4 + // cell
      3 * 4 + // dir
      1 * 4 + // group
      0;

    this.obstacleInstanceSize =
      3 * 4 + // position
      1 * 4 + // rotation-y
      3 * 4 + // scale
      1 * 4 + // padding
      0;

    this.simulationUBOBufferSize =
      1 * 4 + // deltaTime
      1 * 4 + // avoidance (int)
      1 * 4 + // numAgents
      1 * 4 + // gridWidth
      1 * 4 + // iteration
      3 * 4 + // padding
      0;

    this.cellInstanceSize = 
      2 * 4 +   // 2 u32 indices per pair of start/end ptrs
      2 * 4 +   // padding
      0;

    this.gridWidth = gridWidth;

    this.initBuffers();

    this.setBindGroupLayout();
  }

  initBuffers(){

    this.numAgents = Math.pow(2, Math.ceil(Math.log2(this.numValidAgents)));

    const agentData = new Float32Array(this.numAgents * this.agentInstanceSize / 4); ;
    const obstacleData = new Float32Array(this.numObstacles * this.obstacleInstanceSize / 4); ;

    switch(this.testScene){
      case TestScene.PROXIMAL:
        this.initProximal(agentData, obstacleData);
        break;
      case TestScene.BOTTLENECK:
        this.initBottleneck(agentData, obstacleData);
        break;
      case TestScene.DENSE:
        this.initDense(agentData, obstacleData);
        break;
      case TestScene.SPARSE:
        this.initSparse(agentData, obstacleData);
        break;
      case TestScene.OBSTACLES:
        this.initObstacles(agentData,obstacleData);
        break;
      case TestScene.CIRCLE:
        this.initCircle(agentData, obstacleData);
        break;
    }

    // simulation parameter buffer
    this.simulationUBOBuffer = this.device.createBuffer({
      size: this.simulationUBOBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // agent buffer
    this.agents1Buffer = this.device.createBuffer({
      size: this.numAgents * this.agentInstanceSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Float32Array(this.agents1Buffer.getMappedRange()).set(agentData);
    this.agents1Buffer.unmap(); 

    this.agents2Buffer = this.device.createBuffer({
      size: this.numAgents * this.agentInstanceSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: false
    });

    // cells buffer
    this.cellsBuffer = this.device.createBuffer({
      size: this.gridWidth * this.gridWidth * this.cellInstanceSize,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: false
    });

    // obstacles buffer
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
        simulationParams.avoidance,
        this.numAgents,
        this.gridWidth
      ])
    );
  }

  setIteration(itr: number){
    this.device.queue.writeBuffer(
      this.simulationUBOBuffer,
      4*4,
      new Int32Array([
        itr
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
        binding: 1, // agents1Buffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
      {
        binding: 2, // agents2Buffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
      {
        binding: 3, // cellsBuffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
      {
        binding: 4, // obstacleBuffer
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      }
    ]
  });
  }

  getBindGroup(flip: boolean = false){
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
            buffer: flip ? this.agents2Buffer : this.agents1Buffer,  // READ
            offset: 0,
            size: this.numAgents * this.agentInstanceSize,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: flip ? this.agents1Buffer : this.agents2Buffer,  // WRITE
            offset: 0,
            size: this.numAgents * this.agentInstanceSize,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.cellsBuffer,
            offset: 0,
            size: this.gridWidth * this.gridWidth * this.cellInstanceSize,
          },
        },
        {
          binding: 4,
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

  setObstacleData(data : Float32Array, index : number, position : number[], rotation : number, scale : number[]) {
    const offset = this.obstacleInstanceSize * index / 4;

    data[offset + 0] = position[0]; //100*(Math.random()-0.5);
    data[offset + 1] = minY;
    data[offset + 2] = position[1];

    data[offset + 3] = rotation; //rotation

    data[offset + 4] = scale[0];
    data[offset + 5] = obstacleHeight / 2.0;
    data[offset + 6] = scale[1];
  }

  setAgentData(
    agents : Float32Array, index : number, position : number[], color : number[], 
    velocity : number[], speed: number, goal : number[]) {
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

    agents[offset + 15] = speed;

    agents[offset + 16] = goal[0];
    agents[offset + 17] = minY;
    agents[offset + 18] = goal[1];

    agents[offset + 20] = goal[0] - position[0];
    agents[offset + 21] = 0;
    agents[offset + 22] = goal[1] - position[1];
  }

  initProximal(agents : Float32Array, obstacles: Float32Array) {
    for (let i = 0; i < this.numAgents/2; ++i) {
      let x = Math.floor(i/10);
      let z = i%10 + 5;
      let v = 0.5;
      this.setAgentData(agents, 2*i, [0.1+x, z], [1,0,0,1], [0,-v], preferredVelocity, [0, -scatterWidth]);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], [0,0,1,1], [0,v], preferredVelocity, [0, scatterWidth]);
    }
  }

  initBottleneck(agents : Float32Array, obstacles: Float32Array) {
    for (let i = 0; i < this.numAgents; ++i) {
      let x = i%20 - 10;
      let z = Math.floor(i/20) + 10;
      let v = 0.5;
      this.setAgentData(
        agents, i,
        [0.1+x, z], [1,0,0,1], [0,-v], preferredVelocity, [0, -scatterWidth]);
    }

    this.setObstacleData(obstacles, 0, [25,-25], 0, [20, 20]);
    this.setObstacleData(obstacles, 1, [-25,-25], 0, [20, 20]);
  }

  initDense(agents : Float32Array, obstacles: Float32Array) {
    for (let i = 0; i < this.numAgents/2; ++i) {
      let x = i%100 - 50;
      let z = Math.floor(i/100) + 10;
      let v = 0.5;
      this.setAgentData(agents, 2*i, [0.1+x, z], [1,0,0,1], [0,-v], preferredVelocity, [0, -scatterWidth]);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], [0,0,1,1], [0,v], preferredVelocity, [0, scatterWidth]);
    }
  }

  initSparse(agents: Float32Array, obstacles: Float32Array) {
    for (let i = 0; i < this.numAgents/2; ++i) {
      let x = 2*(i%100) - 50;
      let z = 2*Math.floor(i/100) + 10;
      let v = 0.5;
      let s = (Math.random() - 0.5) + preferredVelocity;
      this.setAgentData(agents, 2*i, [0.1+x, z], [1,0,0,1], [0,-v], s, [0, -scatterWidth]);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], [0,0,1,1], [0,v], s, [0, scatterWidth]);
    }
  }

  initObstacles(agents: Float32Array, obstacles: Float32Array) {
    for (let i = 0; i < this.numAgents/2; ++i) {
      let x = i%100 - 50;
      let z = Math.floor(i/100) + 10;
      let v = 0.5;
      this.setAgentData(agents, 2*i, [0.1+x, z], [1,0,0,1], [0,-v], preferredVelocity, [0, -scatterWidth]);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], [0,0,1,1], [0,v], preferredVelocity, [0, scatterWidth]);
    }

    for (let i = 0; i < obstacles.length; i++)
    {
      
      let scale = Math.random() * 3 + 1;
      let rot = Math.random() * Math.PI;
      this.setObstacleData(obstacles, i, [(Math.random()-0.5)*scatterWidth,0], rot, [scale, scale]);
    }
  }

  initCircle(agents: Float32Array, obstacles: Float32Array) {

    let radius = this.numAgents * diskRadius / Math.PI;
    
    for(let i = 0; i < this.numAgents; i++) {
      let t = (i/this.numAgents) * 2.0 * Math.PI;
      let x = radius * Math.cos(t);
      let z = radius * Math.sin(t);
      let c = [Math.random(), Math.random(), Math.random(), 1];
      let s = (Math.random() - 0.5) + preferredVelocity;
      this.setAgentData(agents, i, [x, z], c, [0, 0], s, [-x,-z]);
    }
  }

}
