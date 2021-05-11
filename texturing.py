import numpy as np
import pickle
import cv2
import os

def texture_cube_mesh(path_to_data, file_name, block_path="MinecraftTextures/block/", helper_path="objs/helper_data/", debug=False):
    obj = Get_OBJ_Data(path_to_data, f'results/{file_name}_cube')
    faces, xyz = get_voxel_mappings(obj)
    texture, block_to_texture = Create_Block_Texture(block_path, helper_path, path_to_data, save=True) #Note, Save=True Needed to create Material once for each data_dir
    cubed_textured_path = Create_Textured_Mesh(file_name, block_to_texture, path_to_data, obj, xyz, helper_path=helper_path)

    if debug:
        print(f"Mesh Created : {cubed_textured_path}")
        # TODO: plot the mesh to debug


def get_voxel_mappings(mesh):

    faces = np.round(mesh['v'][(mesh['f'] - 1)[:]])

    v1 = faces[:, 1] - faces[:, 0]
    v2 = faces[:, 2] - faces[:, 0]

    norm = np.cross(v1, v2)
    midpoint = (faces[:, 1] - faces[:, 2]) /2 + faces[:, 2] - norm / 2

    xyz = midpoint.astype('int32')

    return faces, xyz


def Get_OBJ_Data(path_to_data, file_name):
    # Dictionary storing OBJ data
    obj = {
    "v": [],
    "f": [],
    "vt": []
    }

    # Parses and reads OBJ
    # Assumes Specific Format
    with open(f"{path_to_data}{file_name}.obj") as f:
        for i, line in enumerate(f):
            tokenized_line = line.split()
            key = tokenized_line[0]
            
            if key in ["mtllib", "usemtl"]:
                # could be important in the future             
                continue
            elif key == "f":
                numbers_pairs = [x.split("/") for x in tokenized_line[1:]]
                obj[key].append(list(map(lambda x: int(x[0]), numbers_pairs)))
            else:
                numbers = [float(x) for x in tokenized_line[1:]]
                obj[key].append(numbers)

    for key,value in obj.items():
        obj[key] = np.array(value)

    return obj

def Create_Block_Texture(block_path, helper_path, path_to_data, load_existing=False, save=False):
    if load_existing:
        with open(f"{helper_path}block_to_texture.pkl", "rb") as f:
            block_to_texture = pickle.load(f)

        texture = cv2.imread(f"{helper_path}all_mc_textures.png")

        return texture, block_to_texture

    MAX_NUM_MINECRAFT_TEXTURES = len(list(os.listdir(block_path)))
    texture_width = 16

    # All_MC_Blocks_Texture
    texture = np.zeros((MAX_NUM_MINECRAFT_TEXTURES * texture_width, texture_width, 3))

    # Maps Block to Texture UV Coord
    block_to_texture = {}

    cur_coord = 0
    step_size = 1 / MAX_NUM_MINECRAFT_TEXTURES
    y = 0
    for file in os.listdir(block_path):
        if not file.endswith(".png"):
            continue
        
        # Temp Hard Code Of Top Right Triangle
        block_to_texture[file] = [[0, 1 - y], [1, 1 - y], [1, 1 - (y + step_size)]]
        
        # Sets Part of Image to MC Block
        image = cv2.imread(f"{block_path}{file}")
        texture[cur_coord:cur_coord + texture_width,:] = image

        y += step_size
        cur_coord += texture_width

    if save:
        with open("block_to_texture.pkl", "wb") as f:
            pickle.dump(f"{helper_path}block_to_texture", f)

        cv2.imwrite(f"{helper_path}all_mc_textures.png", texture)
        cv2.imwrite(f"{path_to_data}results/all_mc_textures.png", texture)

    return texture, block_to_texture

def Create_Textured_Mesh(file_name, block_to_texture, path_to_data, obj, xyz, helper_path):
    # Loads Voxel to Block Dictionary
    with open(f"{path_to_data}results/voxel_to_block_{file_name}", "rb") as f:
        voxel_to_block = pickle.load(f)

    # Reader Helper Data
    with open(f"{helper_path}FORMAT.mtl", "r") as f:
        mtl_text = f.read().replace("MESH", file_name)

    # Lines List Containing Text For OBJ
    vt_lines = []
    face_lines = []

    ft_idx = 1
    for f, voxel in zip(obj["f"], xyz):
        # From Voxel, Get Its Respective Block
        block = voxel_to_block[tuple(voxel)]
        
        # Appends Vt data
        vt = block_to_texture[block]
        vt_lines.append("\n".join(["vt " + x for x in map(lambda v: " ".join([str(y) for y in v]), vt)]))
    
        # Append Face Data
        line = "f "
        for i in range(3):
            line += f"{f[i]}/{ft_idx} "
            ft_idx += 1
        
        face_lines.append(line)

    cubed_textured_path = f"{path_to_data}/results/{file_name}_cube_textured"
    
    # Write Material
    with open(f"{cubed_textured_path}.mtl", "w") as f:
        f.write(mtl_text)

    # Write to Output
    with open(f"{cubed_textured_path}.obj", "w") as f:
        contents = "mtllib mesh_1.mtl\n"
        contents += "\n".join(["v " + x for x in map(lambda v: " ".join([str(y) for y in v]), obj["v"])])
        contents += "\n"
        contents += "\n".join(vt_lines)
        contents += "\n"
        contents += "usemtl mesh_1" + "\n"
        contents += "\n".join(face_lines)
        f.write(contents)

    return f"{cubed_textured_path}.obj"