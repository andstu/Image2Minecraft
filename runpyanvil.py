#!/bin/python3
import sys
from pyanvil import World, BlockState, Material
import numpy as np


def generate_in_world(world_name, saves_folder, npy_voxels, block_type=Material.gold_block, offset_pos=(5, 5, 5)):
    with World(world_name, save_location=saves_folder, debug=False) as wrld:
        if isinstance(npy_voxels, str) and npy_voxels[-4:] == '.npy':
            obj = np.load(npy_voxels)
        elif isinstance(npy_voxels, np.ndarray):
            obj = npy_voxels
        else:
            raise TypeError("npy_voxels is not set to a usable type.")
        obj = obj.squeeze(0)
        voxels = np.argwhere(obj == 1)
        voxels[:, [1, 2]] = voxels[:, [2, 1]]

        for voxel in voxels:
            wrld.get_block(list(voxel + offset_pos)).set_state(BlockState(block_type))

    print("Done!")
        

generate_in_world("Flat", 'C:/Users/gsmel/AppData/Roaming/.minecraft/saves', 'toilet.npy')


# with World('Flat', save_location='C:/Users/gsmel/AppData/Roaming/.minecraft/saves', debug=True) as wrld:
#     cv = wrld.get_canvas()

#     cv.select_rectangle((-100, 4, -100), (100, 31, 100)).fill(BlockState(Material.air, {}))
#     # cv.select_rectangle((-1, 4, -1), (1, 48, 1)).fill(BlockState(Material.diamond_block, {}))
#     # cv.select_rectangle((-100, 3, -100), (100, 3, 100)).fill(BlockState(Material.grass_block, {}))

#     # path = []
#     # loc = (0, 0)
#     # dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     # direction = 0
#     # for step in range(500):
#     #     path.append((loc[0], 4, loc[1]))
#     #     loc = tuple(map(lambda a, b: a + b, loc, dirs[direction]))
        
#     #     if random.randrange(100) % 20 == 0:
#     #         # print("test")
#     #         direction = (direction + 1) % len(dirs)

    
#     # for block in path:
#     #     # cv.select_rectangle(block, block).fill(BlockState(Material.obsidian, {}))
#     #     print(block)
#     #     myBlock = wrld.get_block(block)
#     #     myBlock.set_state(BlockState(Material.obsidian, {}))

#     toilet = np.load('toilet.npy')
#     toilet = toilet.squeeze(0)
#     voxels = np.argwhere(toilet == 1)
#     voxels[:, [1, 2]] = voxels[:, [2, 1]]

#     for voxel in voxels:
#         # list(voxel + 4)
#         wrld.get_block(list(voxel + 4)).set_state(BlockState(Material.gold_block))

# print("Saved!")