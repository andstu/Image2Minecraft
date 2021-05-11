import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt


def compare_images(file_name, img_dir='results/'):
    pass

def compute_IOU(path_to_data, file_name):
    pass

def render_mesh(path_to_data, file_name, output_dir='results/'):

    mesh_file = "objs/cub/mesh_1.obj"



    tmesh = trimesh.load_mesh(mesh_file)
    trimesh.repair.fix_normals(tmesh)

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


        t = [0, 0, 2.5]
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

        light = pyrender.SpotLight(color=np.ones(3), intensity=13.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(640, 480)
        color, depth = r.render(scene)
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.imshow(color)

        plt.savefig(f'test_imgs/{theta_deg}.png')

# plt.show()