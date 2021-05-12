# Image2Minecraft

## Overview

In this project we created a streamlined pipeline for converting a 2D image into a 3D representation in the popular voxel-based video game, Minecraft. Our project leverages Nikola Zubic and Pietro Lio's paper (found here: https://github.com/NikolaZubic/2dimageto3dmodel) on converting from a single image to a 3D model to get a 3D reconstruction of the image then generates a voxel construction of that model after which textures are selected from the minecraft blocks. The model is then rendered and the 3D reconstruction is compared to a render of the voxel recreation with Minecraft block textures. When the images are similar enough the object can be imported into a Minecraft world using our fork of DanaO's PyAnvilEditor (found here: https://github.com/IsItGreg/PyAnvilEditor).

## Running the Code
`mkdir results`

`mkdir results/p3d`

`mkdir results/cub`

`pip install requirements.txt`

`python testing.py`

This code can be with the pre generated meshes in `objs/` or can be run with any other obj file.

In order to generate the meshes from 2D images please refer to the instructions at [2dimageto3dmodel](https://github.com/NikolaZubic/2dimageto3dmodel).

## Our Contributions

Andrew Teeter:
- Implemented the testing.py script
- Implemented lines 1 - 60 in texturing.py
- Implemented lines 236 - 280 in generator.py 

Matthew Clinton:
- Implemented optimize.py
- Implemented lines 1 - 236 in generator.py
- Implemented lines 60 - 153 in texturing.py
- Implemented run.py

Gregory Smelkov:
- Forked PyAnvilEditor and fixed a bug that would crash it when blocks are added to sections without blocks
- Created the runpyanvil.py script with functions to take in a 3D array of voxel positions and an array of block types and generate the model within Minecraft (whole file)
- Created the render_3d_test.py script including figuring out how to actually get a render of the model from various angles which was eventually copied into testing.py and modified slightly (whole file)
