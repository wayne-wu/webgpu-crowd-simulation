import { mat4, vec3, vec2 } from "gl-matrix";

const scatterWidth = 45;
const diskRadius = 0.5;
const invMass = 0.5;
const minY = 0.5;
const obstacleHeight = 3.0;
const preferredVelocity = 1.4;

const agentColor1 = [(117 + (100 * Math.random() - 50)) / 255.0, 4 / 255.0, (60 + (50 * Math.random() - 25)) / 255.0, 1];
const agentColor2 = [17 / 255.0, (113 + (50 * Math.random() - 25)) / 255.0, ((Math.random() * 200 - 100) + 128) / 255.0, 1];

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
  numGoals    : number;

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
  goalsBuffer : GPUBuffer;

  goalData : Array<vec2>;             // goal positions for rendering
  
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
    this.goalData = new Array<vec2>();

    this.agentPositionOffset = 0;
    this.agentColorOffset = 4 * 4;
    this.agentVelocityOffset = 20 * 4;

    this.numAgents = Math.pow(2, Math.ceil(Math.log2(numAgents)));
    this.numValidAgents = numAgents;
    
    this.numObstacles = 1;
    this.numGoals = 0;

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

    const agentData = new Float32Array(this.numAgents * this.agentInstanceSize / 4); 
    const obstacleData = new Float32Array(this.numObstacles * this.obstacleInstanceSize / 4); 

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

    const goalsArray = this.getGoalsArray();

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

    // goals buffer
    this.goalsBuffer = this.device.createBuffer({
      size: this.goalData.length * 6*4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    new Float32Array(this.goalsBuffer.getMappedRange()).set(goalsArray);
    this.goalsBuffer.unmap();

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

  getGoalsArray(){
    let tmpGoalVertArray = [];
    for (let i = 0; i < this.goalData.length; i++){
      tmpGoalVertArray[i * 6 + 0] = this.goalData[i][0];
      tmpGoalVertArray[i * 6 + 1] = 1.0;
      tmpGoalVertArray[i * 6 + 2] = this.goalData[i][1];
      tmpGoalVertArray[i * 6 + 3] = 1.0;
      tmpGoalVertArray[i * 6 + 4] = 1.0;  // filler data
      tmpGoalVertArray[i * 6 + 5] = 1.0;  // filler data
    }
    this.numGoals = this.goalData.length;
    return new Float32Array(tmpGoalVertArray);
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
    const computeBindGroup = this.device.createBindGroup({
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
    velocity : number[], speed: number, goal : number[], group: number = 0) {
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

    agents[offset + 23] = group;
  }

  initProximal(agents : Float32Array, obstacles: Float32Array) {
    const planeWidth = 25; // plane width is 30 -- place 5 before that
    const tmpGoalData = new Array<vec2>();
    for (let i = 0; i < this.numAgents/2; ++i) {
      let x = Math.floor(i/10);
      let z = i%10 + 5;
      x *= 1.1;
      z *= 1.1;
      const v = 0.5;
      this.setAgentData(agents, 2*i, [0.15+x, z], agentColor1, [0,-v], preferredVelocity, [0, -planeWidth], 0);
      this.setAgentData(agents, 2*i + 1, [-0.15+x, -z], agentColor2, [0,v], preferredVelocity, [0, planeWidth], 1);
    }
    tmpGoalData.push(vec2.fromValues(0, -planeWidth));
    tmpGoalData.push(vec2.fromValues(0, planeWidth));
    this.goalData = tmpGoalData;
  }

  initBottleneck(agents : Float32Array, obstacles: Float32Array) {
    const tmpGoalData = new Array<vec2>();
    for (let i = 0; i < this.numAgents; ++i) {
      const x = 1.5*(i%10) - 8;
      const z = 1.1*Math.floor(i/10) + 10;
      const v = 0.5;
      this.setAgentData(
        agents, i,
        [0.1+x, z], agentColor1, [0,-v], preferredVelocity, [0, -scatterWidth]);
    }
    tmpGoalData.push(vec2.fromValues(0, -scatterWidth));
    this.goalData = tmpGoalData;

    this.setObstacleData(obstacles, 0, [23,-25], 0, [20, 20]);
    this.setObstacleData(obstacles, 1, [-23,-25], 0, [20, 20]);
  }

  initDense(agents : Float32Array, obstacles: Float32Array) {
    const planeWidth = 95;  // platform width is 100 -- place 5 before end
    const tmpGoalData = new Array<vec2>();
    for (let i = 0; i < this.numAgents/2; ++i) {
      const x = i%200 - 100;
      const z = Math.floor(i/200) + 10;
      const v = 0.5;
      this.setAgentData(agents, 2*i, [0.1+x, z], agentColor1, [0,-v], preferredVelocity, [0, -planeWidth], 0);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], agentColor2, [0,v], preferredVelocity, [0, planeWidth], 1);
    }

    tmpGoalData.push(vec2.fromValues(0, -planeWidth));
    tmpGoalData.push(vec2.fromValues(0, planeWidth));
    this.goalData = tmpGoalData;
  }

  initSparse(agents: Float32Array, obstacles: Float32Array) {
    const planeWidth = 95;  // platform width is 100 -- place 5 before end
    const tmpGoalData = new Array<vec2>();
    for (let i = 0; i < this.numAgents/2; ++i) {
      const x = 2*(i%100) - 2*50;
      const z = 2*Math.floor(i/100) + 10;
      const v = 0.5;
      const s = (Math.random() - 0.5) + preferredVelocity;
      this.setAgentData(agents, 2*i, [0.1+x, z], agentColor1, [0,-v], s, [0, -planeWidth], 0);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], agentColor2, [0,v], s, [0, planeWidth], 1);
    }

    tmpGoalData.push(vec2.fromValues(0, -planeWidth));
    tmpGoalData.push(vec2.fromValues(0, planeWidth));
    this.goalData = tmpGoalData;
  }

  initObstacles(agents: Float32Array, obstacles: Float32Array) {
    const tmpGoalData = new Array<vec2>();
    for (let i = 0; i < this.numAgents/2; ++i) {
      const x = i%100 - 50;
      const z = Math.floor(i/100) + 10;
      const v = 0.5;
      this.setAgentData(agents, 2*i, [0.1+x, z], agentColor1, [0,-v], preferredVelocity, [0, -scatterWidth], 0);
      this.setAgentData(agents, 2*i + 1, [-0.1+x, -z], agentColor2, [0,v], preferredVelocity, [0, scatterWidth], 1);
    }

    for (let i = 0; i < obstacles.length; i++)
    {
      
      const scale = Math.random() * 3 + 1;
      const rot = Math.random() * Math.PI;
      this.setObstacleData(obstacles, i, [(Math.random()-0.5)*scatterWidth,0], rot, [scale, scale]);
    }

    tmpGoalData.push(vec2.fromValues(0, -scatterWidth));
    tmpGoalData.push(vec2.fromValues(0, scatterWidth));
    this.goalData = tmpGoalData;
  }

  initCircle(agents: Float32Array, obstacles: Float32Array) {
    const tmpGoalData = new Array<vec2>();
    const radius = this.numAgents * diskRadius / Math.PI;
    
    for(let i = 0; i < this.numAgents; i++) {
      const t = (i/this.numAgents) * 2.0 * Math.PI;
      const x = radius * Math.cos(t);
      const z = radius * Math.sin(t);
      const c = [Math.random(), Math.random(), Math.random(), 1];
      const s = (Math.random() - 0.5) + preferredVelocity;
      this.setAgentData(agents, i, [x, z], c, [0, 0], s, [-x,-z], i);
      tmpGoalData.push(vec2.fromValues(-x, -z));
    }
    this.goalData = tmpGoalData;
  }

}
