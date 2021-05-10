import open3d as o3d
from open3d import geometry
from open3d import utility

mesh = o3d.io.read_triangle_mesh("objs/cub/mesh_0.obj")
o3d.visualization.draw_geometries([mesh])

# mesh1.paint_uniform_color([1, 0.706, 0])