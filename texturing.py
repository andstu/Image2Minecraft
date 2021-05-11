import numpy as np

def texture_cube_mesh(path_to_data, file_name, resolution):
    obj = Get_OBJ_Data('objs/p3d/results/', 'mesh_1_cube')
    faces, xyz = get_voxel_mappings(obj)


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
    "f": []
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

