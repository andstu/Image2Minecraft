#!/bin/python3
import sys
from pyanvil import World, BlockState, Material
import numpy as np
import pickle


def generate_in_world(world_name, saves_folder, npy_voxels, block_types=Material.gold_block, offset_pos=(-50, 5, -50)):
    with World(world_name, save_location=saves_folder, debug=False) as wrld:
        if isinstance(block_types, str):
            with open(block_types, 'rb') as f:
                block_types = pickle.load(f)
        
        
        if isinstance(npy_voxels, str) and npy_voxels[-4:] == '.npy':
            obj = np.load(npy_voxels)
        elif isinstance(npy_voxels, np.ndarray):
            obj = npy_voxels
        else:
            raise TypeError("npy_voxels is not set to a usable type.")
#         obj = obj.squeeze(0)
        voxels = np.argwhere(obj == 1)
#         voxels[:, [1, 2]] = voxels[:, [2, 1]]
        
        for voxel in voxels:
#             print('minecraft:' + block_types[tuple(voxel[[0,2, 1]])][:-4])
#             wrld.get_block(list(voxel + offset_pos)).set_state(BlockState('minecraft:' + block_types[tuple(voxel[[0,2, 1]])][:-4]))
            wrld.get_block(list(voxel + offset_pos)).set_state(BlockState('minecraft:' + block_types[tuple(voxel)][:-4]))
            
            test = 'wet_sponge.png'
#             wrld.get_block(list(voxel + offset_pos)).set_state(BlockState('minecraft:' + test[:-4]))


    print("Done!")
        

# save_path = r'C:\Users\atlig\AppData\Roaming\.minecraft\saves'
save_path = r'/home/mfclinton/.minecraft/saves'

dataset = "p3d"
path_to_data = f"objs/{dataset}/"
file_name = 'mesh_0' # Assumes mtl and png have the same name
offset_pos=(450, 5, 450)
generate_in_world("Flat", save_path, f'Material_Extraction/{path_to_data}results/cube_world_{file_name}.npy', block_types=f'Material_Extraction/{path_to_data}results/voxel_to_block_{file_name}',offset_pos=offset_pos)


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