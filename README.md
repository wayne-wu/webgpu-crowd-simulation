# Real-Time Crowd Simulation in WebGPU

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

by [Ashley Alexander-Lee](), [Matt Elser](), and [Wayne Wu](https://www.wuwayne.com/).

[**Check it out here! (WebGPU Required)**](https://www.wuwayne.com/webgpu-crowd-simulation/)

<img width="974" alt="Screen Shot 2021-12-12 at 11 57 46 AM" src="https://user-images.githubusercontent.com/77313916/145721748-dc58ae6a-2659-462c-9ea9-d2c527b6714f.png">

Installation
============
1. Clone this repo
2. Run `npm i`
3. Build with `npm run-script build`
4. Start with `npm start`
5. Must view using Google Chrome Canary or Chrome Developer
6. Be sure to `--enable-unsafe-webgpu` in the Chrome Settings

Usage
=====
**Camera Controls**
- `Left Mouse Hold and Move`: change camera orientation
- `Right Mouse Hold and Move`: pan camera
- `Scroll in/out`: zoom in/out
- `resetCamera`: gui control to reset camera to the scene's default

**Grid Controls**
- `gridWidth`: change the number of divisions of the hash grid (hash grid size is constant)
- `gridOn`: turn the grid checkerboard visualization on or off

**Scene Controls**
- `scene`: choose the scene configuration, which changes the number of agents, agent goals, camera default, and platform width
- `model`: choose which model to use
- `showGoals`: show where the agent goals are (marked with rainbow spheres)
- `shadowOn`: show the agent shadows
- `agent selector`: choose the number of agents in powers of 2 (only supported in dense and sparse scenes)

**Simulation Controls**
- `simulate`: start/stop simulation
- `deltaTime`: the time step to use in the simulation
- `lookAhead`: parameter that affects how far the agents look ahead to plan their trajectory
- `avoidanceModel`: use the avoidance model, which allows agents to move in a more tangential direction when stuck in a dense area
- `resetSimulation`: reset the simulation


Overview
===========
This project attempts to implement a real-time crowd simulation based on the paper: [Position-Based Real-Time Simulation of Large Crowds](https://tomerwei.github.io/pdfs/mig2017.pdf). 
Unlike the paper which uses CUDA and Unreal Engine for simulation and rendering,
this project uses WebGPU for both.

![Real-Time Crowd Simulation GIF](img/obstacle_example.gif)

## Neighbor Searching
In order to ensure our neighbor finding was efficient, we employed a hash grid neighbor finder, based on [this article](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf). 

![hash grid debugging](/img/grid_cell_assignment_vis.png)
*Pictured above: A visualization of the early cell assignments, used to debug our initial implementation*

## Position-Based Dynamics
The main solver used in the paper is Position-based Dynamics with Jacobi Solver, which can be parallelized very easily.

### Short Range Collision
The first constraint applied is a simple collision constraint model for short range collision.
This resolves the immediate collisions between neighboring agents to prevent penetration.

![Short Range](img/shortrange.gif)

### Long Range Collision
Long range collision constraint is used to enable agents to look ahead in the future for possible collisions.
The constraint will predict the position of neighboring agents at a specified future time and resolve any collision at the future position.
As shown in the image below, the agents start reacting before they are even close to colliding. 
User can tweak the **lookAhead** parameter to specify how far ahead to an agent should look ahead for long range collision.

lookAhead = 6         | lookAhead = 12                         
:--------------------:|:------------------:
![](img/longrange_t6.gif)        |![](img/longrange_t12.gif)   

### Long Range Collision w/ Avoidance Model
The paper introduces a novel addition to the long range collision constraint that prevents agents from being pushed back, typically in a dense crowd.
The avoidance model considers only the tangential component of the position correction, 
thus removing/reducing the correction along the contact normal (which can push the agent back if two agents are walking towards each other).

Scene         | LR                 |  LR w/ Avoidance                 
:------------:|:------------------:|:-------------------------:
Proximal      |![](img/proximal_longrange.gif)   |  ![](img/proximal_avoidance.gif)     
Dense         |![](img/dense_longrange.gif)   |  ![](img/dense_avoidance.gif)

### Frictional Contact

### Cohesion
Cohesion is added so that agents in the same group will tend to follow each other thus creating smoother motions.

![cohesion](img/cohesion_debug.gif)

### Obstacles Collision and Avoidance
To add complexity to the scene, we support box-shaped obstacles in our implmentation.
The paper showcases walls as obstacles which can be modeled as line segments with short range collision constraint. 
We use a similar approach that considers each edge of the box as a wall constraint.

Furthermore, to enable agents to look ahead in the future and avoid obstacles, 
we implement an obstacle avoidance model based on [OpenSteer](http://opensteer.sourceforge.net). 
This affects the velocity directly in the final stage (similar to cohesion) instead of correcting the position using constraint projection.
While this approach is not specifically outlined in the paper, we suspect the author having something similar based on the result produced.

Obstacles         | Bottleneck                         
:----------------:|:------------------:
![](img/obstacles.gif)        |![](img/bottleneck.gif)    

### Parameters Tuning
The author has kindly provided the parameters used in the paper. Using it as a starting point has given us a reasonable result.

## Rendering
### Model Loading
We support several different models in order to produce a visually compelling scene, which you can select via the "models" dropdown in the gui. We used the `GLTFLoader` from [threejs](https://threejs.org/docs/#examples/en/loaders/GLTFLoader) to parse our gltf files, and we use the resulting `gltf` object to create the array buffer that we use in the WebGPU rendering pipeline. Each of the models affects the FPS proportionally to model complexity, with the duck model taking the least time, and the xbot taking the most (TODO: chart showing FPS impact of each model). We were able to borrow the gltf models from the following sources:
- 'Duck' from [threejs](https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf)
- 'Cesium Man' from [Cesium](https://github.com/CesiumGS/cesium)
- 'XBot' from [Mixamo](https://www.mixamo.com/)
- 'Ship', 'Tower', 'Garbage Truck', 'Sedan' from [Kenney Assets](https://kenney.nl/assets?q=3d)

| ![model example1](/img/model_example1.png) | ![model example2](/img/model_example2.png) |
| ------------------------------------------ | ------------------------------------------ |
| ![model_example3](/img/model_example3.png) | ![model example4](/img/model_example4.png) |

## Test Scenes

## Future Work
* Animation & Skinning
* Cascaded Shadow Mapping
* Separate Hash Grids for Short Range vs. Long Range

References
==========
- Base code from this [Particles WebGPU sample](https://github.com/austinEng/webgpu-samples) by Austin Eng
- [3d-view-controls](https://www.npmjs.com/package/3d-view-controls) for camera manipulation
- Camera class referenced from UPenn's CIS566 base code, written by Adam Mally
- [dat-gui](https://github.com/dataarts/dat.gui) for gui controls

Bloopers
========
### Upside-down Agents
![upside down agents](/img/blooper3.gif)

### Giant Agents
![giant agents](/img/blooper1.gif)
