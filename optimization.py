import numpy as np
import torch
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
from testing import *
from generator import generate, Get_Minecraft_Textures
from texturing import texture_cube_mesh
import black_box as bb
import pickle
import time
import cma

block_path = "MinecraftTextures/block/"
resolution = 50
metric = "w_eucl"
filter_list = Create_Filter_List(block_path)

dataset = "p3d"
file_name = "mesh_1"
path_to_data = f'objs/{dataset}/'
cube_path = f'objs/{dataset}/results/'
cube_file = f'{file_name}_cube_textured'

block_names, _ = Get_Minecraft_Textures(block_path, filter_list)

print(f"Num Blocks : ", len(block_names))

# Initializes the Voxel_To_Block
generate(path_to_data, file_name, block_path, resolution, metric, filter_list=filter_list)

# Renders Initial Pictures Of OG Image
rendered_og = render_mesh(path_to_data, file_name, dataset, save=False)

with open(f"{path_to_data}/results/voxel_to_block_{file_name}", "rb") as f:
    voxel_to_block = pickle.load(f)

valid_voxels = list(voxel_to_block.keys())
print(f"Num Voxels : ", len(valid_voxels))

def bb_fun(block_weights):
    block_indexes = np.clip(np.rint(block_weights).astype(int), 0, len(block_names) - 1)
    
    voxel_to_block = {}
    
    for i, voxel in enumerate(valid_voxels):
        block = block_names[block_indexes[i]]
        voxel_to_block[voxel] = block
        
    with open(f"{path_to_data}/results/voxel_to_block_{file_name}", "wb") as f:
        pickle.dump(voxel_to_block, f)
        
    
    # Compute Optimization Metric
    texture_cube_mesh(path_to_data, file_name, save=True)
    rendered_cm = render_mesh(cube_path, cube_file, dataset, save=True)
    score = compare_images_memory(rendered_og, rendered_cm)
    print("Score: ", score)
    
    return - score
    
# From https://github.com/paulknysh/blackbox

# domain = [[0., len(block_names) - 1]] * len(valid_voxels)

es = cma.CMAEvolutionStrategy(len(valid_voxels) * [len(block_names) / 2], len(block_names) / 2)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [bb_fun(x) for x in solutions])


    

# best_params = bb.search_min(f = bb_fun,
#                             domain = domain,
#                             budget = 10000,  # total number of function calls available
#                             resfile = 'optim_output.csv')  # text file where results will be saved