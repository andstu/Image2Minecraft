from generator import generate
from runpyanvil import generate_in_world
import os

def Create_Filter_List(block_path):
    filter_list = []
    for file in os.listdir(block_path):
        if not file.endswith(".png"):
            continue
        
        block = file.replace(".png", "")

        if block.endswith("powder") or block.endswith("leaves") or block == "ladder" or block == "ice" or block.endswith("glass") or block == "snow":
            filter_list.append(file)

    return filter_list
        

if __name__=="__main__":
    # Generation Arguments
    dataset = "p3d"
    path_to_data = f"objs/{dataset}/"
    file_name = 'mesh_1' # Assumes mtl and png have the same name
    block_path = "MinecraftTextures/block/"
    resolution = 50
    metric = "w_eucl"
    filter_list = Create_Filter_List(block_path)
    world_name = "JoeMama"

    # Minecraft Params
    save_path = r'/home/mfclinton/.minecraft/saves'
    offset_pos=(0, 5, 0)
    

    # Generates Cube_World and Texture Mappings
    generate(path_to_data, file_name, block_path, resolution, metric, filter_list=filter_list)
    
    # Creates Minecraft Save 
    # generate_in_world(world_name, save_path, f'{path_to_data}results/cube_world_{file_name}.npy', block_types=f'{path_to_data}results/voxel_to_block_{file_name}',offset_pos=offset_pos)
