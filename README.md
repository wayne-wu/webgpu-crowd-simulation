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

Overview
===========
This project attempts to implement a real-time crowd simulation based on the paper: [Position-Based Real-Time Simulation of Large Crowds](https://tomerwei.github.io/pdfs/mig2017.pdf). 
Unlike the paper which uses CUDA and Unreal Engine for simulation and rendering,
this project uses WebGPU for both.

![Real-Time Crowd Simulation GIF](img/obstacle_example.gif)

## Neighbor Searching
#### Hash Grid Implementation
Each agent reacts to other agents within two radii: near radius and far radius. A naive solution
to finding agents within these radii would be to iterate over all agents in the scene and check their distance. Instead, a hash grid is used. A uniform grid is (non-visually) overlayed onto the plane of agents. The following series of compute shaders turns this conceptual grid into a useful tool, entirely on the GPU:
- For each agent, a thread on the GPU identifies the cell that agent belongs to. Agent's outside the finite hash grid are considered to be in an invalid cell and do not contribute to further computation/simulation, but are rendered grey for clarity. 
- The agent buffer is then sorted based on these cells using a GPU based bitonic sort via a series of compute shaders. This is done in multiple shader calls as a device-wide join is needed after each step. 
- Finally an additional compute shader determines where in the agent buffer each cell's agents starts and ends, storing that data in a cell buffer. For example, agents in cell 33 could occupy indicies X through Y, and agents in cell 34 could then be found at indicies Y+1 through Z.
Having completed all this, all that is needed to find neighbors is to simply iterate over agents between the start and end for the relevant cells. The relevant cells in the paper's implementation are hardcoded to be 9 cells: the agent's current cell and all cells adjacent to it. Our implementation, however calculates which cells are within the relevant radius. This calculation not only makes our hash grid more efficient for larger/denser grids (by ignoring cells outside the agent's radius), it is more robust for smaller cell sizes where the agent's radius may reach farther than just the adjoining cells. 
#### Performance gains
Our implementation can emulate a non-hashgrid implementation by setting the `gridWidth` parameter to 1 (to set the grid to be 1 cell wide by 1 cell long). 


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

![Long Range](img/longrange.gif)

### Long Range Collision w/ Avoidance Model
The paper introduces a novel addition to the long range collision constraint that prevents agents from being pushed back, typically in a dense crowd.
The avoidance model considers only the tangential component of the position correction, 
thus removing/reducing the correction along the contact normal (which can push the agent back if two agents are walking towards each other).

TODO: GIFs of LR vs Avoidance (Proximal)

TODO: GIFs of LR vs Avoidance (Dense)

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

![Obstacle Avoidance](img/obstacle_example.gif)

### Parameters Tuning
The author has kindly provided the parameters used in the paper. Using it as a starting point has given us a reasonable result.

## Rendering

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
