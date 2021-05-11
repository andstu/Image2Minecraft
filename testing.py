import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import norm
import glob
from generator import generate
from texturing import texture_cube_mesh
import os

def compare_images(file_name, img_dir='results/'):

    norms = []
    simms = []

    # print(f'{img_dir}*{file_name}.png')

    for f in glob.glob(f'{img_dir}*{file_name}.png'):
        theta = f[len(img_dir):-(len(file_name) + 5)]
        x = np.asarray(Image.open(f))
        y = np.asarray(Image.open(f'{img_dir}{theta}_{file_name}_cube_textured.png'))
        diff_percent = compare(x, y)
        simms.append(1 - diff_percent)
        # n = norm(x - y, axis=2)
        # norms.append(n.mean())

    # return np.mean(norms)
    return np.mean(simms)

def compare_images_memory(rendered_x, rendered_y, metric = None):

    norms = []
    scores = []

    for i in range(len(rendered_x)):
        x = rendered_x[i]
        y = rendered_y[i]

        score = compare(x, y)
        scores.append(score)

    # return np.mean(norms)
    return np.mean(scores)


def compare(x, y, metric=None):
    score = 0
    
    if metric == None:
        score = 1 - np.sum(np.abs(x - y)) / (255 * x.shape[0] * x.shape[1])
    elif metric == "eucl":
        score = norm(x - y, axis=2).mean()

    return score


def compute_IOU(path_to_data, file_name):
    orig = trimesh.load(f'{path_to_data}{file_name}.obj')
    cube = trimesh.load(f'{path_to_data}results/{file_name}_cube.obj')

    cube.vertices /= cube.scale
    orig.vertices /= orig.scale

    cube.vertices -= cube.center_mass
    orig.vertices -= orig.center_mass

    cube.fix_normals()
    orig.fix_normals()

    cube.fill_holes()
    orig.fill_holes()

    i = cube.intersection(orig).volume
    u = cube.union(orig).volume

    return i/u


def render_mesh(path_to_data, file_name, dataset, output_dir='results/', save=True):

    mesh_file = f"{path_to_data}{file_name}.obj"

    tmesh = trimesh.load_mesh(mesh_file)
    trimesh.repair.fix_normals(tmesh)
    tmesh.vertices /= tmesh.scale
    tmesh.vertices -= tmesh.center_mass

    rendered_pics = []
        # theta = 270
    for theta_deg in range(0, 360, 45):
        theta = 2*np.pi * theta_deg / 360
        pose = np.array([
                [np.cos(theta), 0, -np.sin(theta), 0],
                [0, 1, 0, 0],
                [np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

        mesh = pyrender.Mesh.from_trimesh(tmesh, poses=pose)
        scene = pyrender.Scene()
        scene.add(mesh)

        # v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

        # while v.is_active:
        #     print(v._camera_node.matrix)

        # camera_pose = np.array([
        #     [1.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        camera_pose = np.array(
        [[1, 0, 1, 0],
        [ 0,  1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0.0,  0.0,  0.0,  1.0]])


        t = [0, 0, .85]
        # t = [1, 0, 2]

        f = t / np.linalg.norm(t)
        left = np.cross([0, 1, 0], f)
        l = left / np.linalg.norm(left)
        u = np.cross(f, l)


        mr = np.array([
            [l[0], l[1], l[2], 0.0],
            [u[0], u[1], u[2], 0.0],
            [f[0], f[1], f[2], 0.0],
            [0, 0, 0, 1]
        ])

        mt = np.array([
            [1, 0, 0, t[0]],
            [0, 1, 0, t[1]],
            [0, 0, 1, t[2]],
            [0, 0, 0, 1]
        ])

        camera_pose = np.matmul(mr, mt)


        scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=8.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(640, 480)
        color, depth = r.render(scene)
        rendered_pics.append(color)
        
        if save:
            plt.figure()
            plt.subplot(1, 1, 1)
            plt.axis('off')
            plt.imshow(color)
            plt.savefig(f'{output_dir}{dataset}/{theta_deg}_{file_name}.png')
            plt.close()
    
    return rendered_pics

    
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

    block_path = "MinecraftTextures/block/"
    resolution = 50
    metric = "w_eucl"
    filter_list = Create_Filter_List(block_path)

    bigger_iou = []
    bigger_dist = []

    for dataset in ['p3d', 'cub']:
        iou_list = []
        dist_list = []

        path_to_data = f'objs/{dataset}/'
        for fn in glob.glob(f'objs/{dataset}/*.obj'):
            file_name = fn[len('objs/p3d/'):-4]
            print(f'------{dataset}/{file_name}------')
            # Generates Cube_World and Texture Mappings
            print("Extracting Mesh Data")
            generate(path_to_data, file_name, block_path, resolution, metric, filter_list=filter_list)

            # Creates the Cube Mesh
            print("Creating Cube Mesh")
            texture_cube_mesh(path_to_data, file_name)

            cube_path = f'objs/{dataset}/results/'

            cube_file = f'{file_name}_cube_textured'

            print("Rendering Meshes")
            render_mesh(path_to_data, file_name, dataset)
            render_mesh(cube_path, cube_file, dataset)

            print("Measuring IOU")
            iou = compute_IOU(path_to_data, file_name)

            print("Comparing Images")
            dist = compare_images(file_name, img_dir=f'results/{dataset}/')

            print(f'IOU: {iou}')
            print(f'Simm: {dist}')

            iou_list.append(iou)
            dist_list.append(dist)

        bigger_dist.append(np.mean(dist_list))
        bigger_iou.append(np.mean(iou_list))


    print(f'Avg IOU for p3d: {bigger_iou[0]}')
    print(f'Avg dist for p3d: {bigger_dist[0]}')
    print(f'Avg IOU for cub: {bigger_iou[1]}')
    print(f'Avg dist for cub: {bigger_dist[1]}')



    # path_to_data = "objs/p3d/"
    # file_name = "mesh_1"
    # print(compute_IOU(path_to_data, file_name))
    # render_mesh(path_to_data, file_name, 'p3d')
    # print(compare_images(file_name, img_dir='results/p3d'))

# plt.show()