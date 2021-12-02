# Real-Time Crowd Simulation in WebGPU

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

by [Ashley Alexander-Lee](), [Matt Elser](), and [Wayne Wu](www.wuwayne.com).

## Overview
Based on: [Position-Based Real-Time Simulation of Large Crowds](https://tomerwei.github.io/pdfs/mig2017.pdf).

![Real-Time Crowd Simulation GIF](img/milestone1_progress.gif)

Installation
============
1. Clone this repo
2. Run `npm i`
3. Build with `npm run-script build`
4. Start with `npm start`
5. Must view using Google Chrome Canary
6. Be sure to `--enable-unsafe-webgpu` in the Chrome Settings

References
==========
- Base code from this [Particles WebGPU sample](https://github.com/austinEng/webgpu-samples) by Austin Eng
- [3d-view-controls](https://www.npmjs.com/package/3d-view-controls) for camera manipulation
- Camera class referenced from UPenn's CIS566 base code, written by Adam Mally
- [dat-gui](https://github.com/dataarts/dat.gui) for gui controls
