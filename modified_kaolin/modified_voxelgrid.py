# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

__all__ = ['voxelgrids_to_cubic_meshes', 'voxelgrids_to_trianglemeshes']

verts_template = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0]
    ],
    dtype=torch.float
)

faces_template = torch.tensor(
    [
        [0, 2, 1, 3],
        [0, 1, 4, 5],
        [0, 4, 2, 6]
    ],
    dtype=torch.int64
)
faces_3x4x3 = verts_template[faces_template]
for i in range(3):
    faces_3x4x3[i, :, (i - 1) % 3] -= 1
    faces_3x4x3[i, :, (i + 1) % 3] -= 1

quad_face = torch.LongTensor([[0, 1, 3, 2]])
kernel = torch.zeros((1, 1, 2, 2, 2))
kernel[..., 0, 0, 0] = -1
kernel[..., 1, 0, 0] = 1
kernels = torch.cat([kernel, kernel.transpose(2, 3), kernel.transpose(2, 4)], 0)  # (3,1,2,2,2)

def voxelgrids_to_cubic_meshes(voxelgrids, is_trimesh=True):
    r"""Convert voxelgrids to meshes by replacing each occupied voxel with a cuboid mesh (unit cube). 

    Each cube has 8 vertices and 6 (for quadmesh) or 12 faces 
    (for triangular mesh). Internal faces are ignored. 
    If `is_trimesh==True`, this function performs the same operation
    as "Cubify" defined in the ICCV 2019 paper "Mesh R-CNN": 
    https://arxiv.org/abs/1906.02739.

    Args:
        voxelgrids (torch.Tensor): binary voxel array with shape (B, X, Y, Z).
        is_trimesh (bool): the outputs are triangular meshes if True. Otherwise quadmeshes are returned.
    Returns:
        (list[torch.Tensor], list[torch.LongTensor]): tuple containing the list of vertices and the list of faces for each mesh.

    Example:
        >>> voxelgrids = torch.ones((1, 1, 1, 1))
        >>> verts, faces = voxelgrids_to_cubic_meshes(voxelgrids)
        >>> verts[0]
        tensor([[0., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.],
                [0., 1., 1.],
                [1., 0., 0.],
                [1., 0., 1.],
                [1., 1., 0.],
                [1., 1., 1.]])
        >>> faces[0]
        tensor([[0, 1, 2],
                [5, 4, 7],
                [0, 4, 1],
                [6, 2, 7],
                [0, 2, 4],
                [3, 1, 7],
                [3, 2, 1],
                [6, 7, 4],
                [5, 1, 4],
                [3, 7, 2],
                [6, 4, 2],
                [5, 7, 1]])
    """
    print("ANDSTU IS FUNNY")
    device = voxelgrids.device
    voxelgrids = voxelgrids.unsqueeze(1)
    batch_size = voxelgrids.shape[0]

    face = quad_face.to(device)

    if device == 'cpu':
        k = kernels.to(device).half()
        voxelgrids = voxelgrids.half()
    else:
        k = kernels.to(device).float()
        voxelgrids = voxelgrids.float()

    conv_results = torch.nn.functional.conv3d(
        voxelgrids, k, padding=1)  # (B, 3, r, r, r)

    indices = torch.nonzero(conv_results.transpose(
        0, 1), as_tuple=True)  # (N, 5)
    dim, batch, loc = indices[0], indices[1], torch.stack(
        indices[2:], -1)  # (N,) , (N, ), (N, 3)
    invert = conv_results.transpose(0, 1)[indices] == -1
    _, counts = torch.unique(dim, sorted=True, return_counts=True)

    faces_loc = (torch.repeat_interleave(faces_3x4x3.to(device), counts, dim=0) +
                 loc.unsqueeze(1).float())  # (N, 4, 3)

    faces_batch = []
    verts_batch = []

    for b in range(batch_size):
        verts = faces_loc[torch.nonzero(batch == b)].view(-1, 3)
        if verts.shape[0] == 0:
            faces_batch.append(torch.zeros((0, 3 if is_trimesh else 4), device=device, dtype=torch.long))
            verts_batch.append(torch.zeros((0, 3), device=device))
            continue
        invert_batch = torch.repeat_interleave(
            invert[batch == b], face.shape[0], dim=0)
        N = verts.shape[0] // 4

        shift = torch.arange(N, device=device).unsqueeze(1) * 4  # (N,1)
        faces = (face.unsqueeze(0) + shift.unsqueeze(1)
                 ).view(-1, face.shape[-1])  # (N, 4) or (2N, 3)
        faces[invert_batch] = torch.flip(faces[invert_batch], [-1])

        if is_trimesh:
            faces = torch.cat(
                [faces[:, [0, 3, 1]], faces[:, [2, 1, 3]]], dim=0)

        verts, v = torch.unique(
            verts, return_inverse=True, dim=0)
        faces = v[faces.reshape(-1)].reshape((-1, 3 if is_trimesh else 4))
        faces_batch.append(faces)
        verts_batch.append(verts)

    return verts_batch, faces_batch