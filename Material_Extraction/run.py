import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import pickle
from sklearn.neighbors import NearestNeighbors
import os
import torch
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids, voxelgrids_to_cubic_meshes
import meshplot as mp

# Relevant Code Here Taken From 
# https://stackoverflow.com/questions/11851342/in-python-how-do-i-voxelize-a-3d-mesh 
import voxelize.voxelize as voxelize

def Get_OBJ_Data(path_to_data, file_name):
    # Dictionary storing OBJ data
    obj = {
    "v": [],
    "vt": [],
    "f": [],
    "ft": []
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
                obj[f"{key}t"].append(list(map(lambda x: int(x[1]), numbers_pairs)))
            else:
                numbers = [float(x) for x in tokenized_line[1:]]
                obj[key].append(numbers)

    for key,value in obj.items():
        obj[key] = np.array(value)

    return obj

# Gets Relevant Voxel Data
def Get_Voxel_Data(obj, resolution):
    # NOTE: Pytorch stuff is legacy and no longer necessary, Fix later

    # Gets Vertices and Faces
    verts = torch.Tensor(obj["v"].copy()).unsqueeze(dim=0)
    faces = torch.from_numpy(obj["f"].copy()) - 1

    # Gets Scale Parameter Similarly to Kaolin (scales to [0,1])
    min_val = torch.min(verts, dim=1)[0]
    origin = min_val
    max_val = torch.max(verts, dim=1)[0]
    scale = torch.max(max_val - origin, dim=1)[0]

    # Creates Empty Dense Voxel Grid
    batch_size = verts.shape[0]
    voxelgrids = torch.zeros((batch_size, resolution, resolution, resolution))

    # Scales Vertices to between 0 and 1
    scaled = (verts - origin.unsqueeze(1)) / scale.view(-1, 1, 1)
    
    # Scales to Fall Into Resolution-Sized Voxel World
    big = scaled * (resolution - 1)


    # The triggered voxels
    voxel = []

    # Maps a voxel to the faces falling into it
    voxel_to_faces = {}

    # Iterates over the faces of our mesh to populate "voxel" and "voxel_to_faces"
    for i, f in enumerate(faces.numpy()):
        triangle = (big).squeeze().numpy()[f]
        vals = voxelize.triangle_voxalize(triangle)
        
        for j in vals:
            voxel_key  = tuple(j)
            voxel_to_faces.setdefault(voxel_key, []).append(i)
            
            if j not in voxel:
                voxel.append(j)
                
        

    return np.array(voxel), voxel_to_faces

# Takes in Dense Triggered Voxel List, Creates Sparse Cube World
def Generate_Cube_World(voxel):
    cube_world = np.zeros((resolution,resolution,resolution))
    cube_world[voxel[:,0],voxel[:,1], voxel[:,2]] = 1
    return cube_world


# Extracts the names and the textures for minecraft blocks
def Get_Minecraft_Textures(block_path, filter_list):
    block_names = []
    block_images = []

    for file in os.listdir(block_path):
        if (not file.endswith(".png")) or (file in filter_list):
            continue
            
        block_image = cv2.imread(f"{block_path}{file}")

        block_images.append(block_image)
        block_names.append(file)

    return block_names, block_images

class VoxelMetric:
    def __init__(self, image, minecraft_texture_data, metric):
        valid_metrics = ["avg", "eucl", "w_eucl"]
        if metric not in valid_metrics:
            raise Exception("Invalid Metric")

        self.image = image
        self.block_names, self.block_images = minecraft_texture_data
        self.metric = metric

    def Create_Mean_NN(self):
        mean_colors = []
        for block_image in self.block_images:
            mean_color = block_image.reshape((-1,3)).mean(axis=0)
            mean_colors.append(mean_color)
        
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(np.array(mean_colors))
        self.neigh = neigh


    # TODO: Make More Efficient Later
    # Note: Masked Array and Masked Value Store Same Info.
    # Masked Array is Sparse, Masked Values is Dense
    def Get_Closest_Block(self, masked_array, masked_values):
        if self.metric == "avg":
            b_id = self.Average_Metric(masked_values)

        elif self.metric.endswith("eucl"):
            weights = None

            # BGR color order weights from https://www.compuphase.com/cmetric.htm
            if self.metric.startswith("w"):
                weights = np.array([3, 4, 2]).reshape(1,1,3)

            b_id = self.Eucl_Method(weights, masked_array)

        block = self.block_names[b_id]
        return block


    # ----------
    # METRICS
    # ----------

    def Average_Metric(self, masked_values):
        if not hasattr(self, "neigh"):
            self.Create_Mean_NN()

        mean_color = masked_values.mean(axis=0)
        dists, block_ids = self.neigh.kneighbors([mean_color])
        b_id = block_ids[0][0]
        return b_id

    def Eucl_Method(self, weights, masked_array):
        if weights is None:
            weights = np.array([1,1,1])

        # Gets Bounding Box Information
        mask_row, mask_col, _ = np.where(masked_array.mask)
        row_bounds = (mask_row.min(), mask_row.max()+1)
        col_bounds = (mask_col.min(), mask_col.max()+1)

        # Crops Image to Bounding Box Around Non-Masked
        sub_image = self.image[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1], :]

        # Distances of Sub Image to Block Images
        dists = []
        for img_idx, block_img in enumerate(self.block_images):
            # Resizes Block Image to be Size of Texture
            resized_img = cv2.resize(block_img, (sub_image.shape[1], sub_image.shape[0]), interpolation = cv2.INTER_AREA)
            
            # Divided by 255 to prevent overflow
            sqred_difference = ((sub_image - resized_img)/255)**2

            dist = np.sum(np.sqrt(sqred_difference * weights))

            dists.append(dist)
        
        # Gets Closest Block ID
        b_id = np.argmin(dists)
        return b_id

# Masks the Image to Only Get textures for Triangles within voxel "voxel_key"
def Get_Masked_Values(image, obj, voxel_key, face_indexes):
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    for f_idx in face_indexes:
        ft = obj["ft"][f_idx]
        
        # Gets UV Coordinates of Respective Face
        uv_coords = np.rint(obj["vt"][ft - 1] * image.shape[0]).astype(int)
        uv_coords = uv_coords.reshape((-1,1,2))

        # Fills in mask for the respective Face_Texture
        cv2.polylines(mask, [uv_coords], isClosed=True, color=(255,255,255), thickness=1)
        cv2.fillPoly(mask, [uv_coords], (255,255,255))
    
    # Creates a Masked Array Containing Only Relevant Face Textures
    masked_array = np.ma.array(image, mask = mask == 255)
    return masked_array

def Get_Voxel_Textures(image, obj, voxel_to_faces, metric_helper):
    # Maps Voxels to their Block Texture
    voxel_to_block = {}
    for voxel_key, face_indexes in voxel_to_faces.items():
        masked_array = Get_Masked_Values(image, obj, voxel_key, face_indexes)
        masked_values = masked_array[masked_array.mask].data.reshape((-1,3))

        block = metric_helper.Get_Closest_Block(masked_array, masked_values)
        voxel_to_block[voxel_key] = block

    return voxel_to_block

def main(path_to_data, file_name, block_path, resolution, metric, filter_list=[]):
    # Gets Raw OBJ DAta
    print("Getting OBJ")
    obj = Get_OBJ_Data(path_to_data, file_name)
    image = cv2.imread(f"{path_to_data}{file_name}.png")
    
    # Gets Voxel Data
    print("Getting Voxel Data")
    voxel, voxel_to_faces = Get_Voxel_Data(obj, resolution)
    cube_world = Generate_Cube_World(voxel)
    
    # Gets Minecraft Data
    print("Getting Minecraft Data")
    minecraft_texture_data = Get_Minecraft_Textures(block_path, filter_list)
    
    # Maps Voxel to Minecraft Block Based On Metric
    print("Mapping Voxel to Minecraft Block")
    metric_helper = VoxelMetric(image, minecraft_texture_data, metric)
    voxel_to_block = Get_Voxel_Textures(image, obj, voxel_to_faces, metric_helper)

    # Creates Cube Mesh
    print("Creating Cube Mesh")
    c_v, c_f = voxelgrids_to_cubic_meshes(torch.from_numpy(cube_world).unsqueeze(dim=0))

    # Saves Data
    with open(f"{path_to_data}/results/voxel_to_block_{file_name}", "wb") as f:
        pickle.dump(voxel_to_block, f)

    with open(f'{path_to_data}/results/cube_world_{file_name}.npy', 'wb') as f:
        np.save(f, cube_world)

    with open(f'{path_to_data}/results/cube_world_{file_name}_v.npy', 'wb') as f:
        np.save(f, c_v[0].cpu().numpy())

    with open(f'{path_to_data}/results/cube_world_{file_name}_f.npy', 'wb') as f:
        np.save(f, c_f[0].cpu().numpy())
    

# Arguments
path_to_data = "objs/cub/"
file_name = 'mesh_0' # Assumes mtl and png have the same name
block_path = "../MinecraftTextures/block/"
resolution = 50
metric = "eucl"
# filter_list = []

main(path_to_data, file_name, block_path, resolution, metric)





    

    

