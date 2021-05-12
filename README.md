# Image2Minecraft

## Overview

In this project we created a streamlined pipeline for converting a 2D image into a 3D representation in the popular voxel-based video game, Minecraft. Our project leverages Nikola Zubic and Pietro Lio's paper (found here: https://github.com/NikolaZubic/2dimageto3dmodel) on converting from a single image to a 3D model to get a 3D reconstruction of the image then generates a voxel construction of that model after which textures are selected from the minecraft blocks. The model is then rendered and the 3D reconstruction is compared to a render of the voxel recreation with Minecraft block textures. When the images are similar enough the object can be imported into a Minecraft world using our fork of DanaO's PyAnvilEditor (found here: https://github.com/IsItGreg/PyAnvilEditor).

