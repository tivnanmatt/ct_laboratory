import torch


def standard_3d_image_transform(n_x, n_y, n_z, s_x, s_y, s_z):

    # 2) Construct an affine transform M, b for (i,j,k)->(x,y,z).
    #    Let's set the center of the volume at i_mid, j_mid, k_mid, 
    #    and voxel size in x,y,z directions = voxel_size.
    i_mid = (n_x - 1) / 2.0
    j_mid = (n_y - 1) / 2.0
    k_mid = (n_z - 1) / 2.0

    # If we assume x = i * voxel_size, y = j * voxel_size, z = k * voxel_size,
    # all orthonormal, then M is diag(voxel_size, voxel_size, voxel_size).
    M = torch.eye(3, dtype=torch.float32)
    M[0, 0] = s_x
    M[1, 1] = s_y
    M[2, 2] = s_z

    # We want the center (i_mid, j_mid, k_mid) to map to (0,0,0).
    # So we set b accordingly:
    b = torch.tensor([-i_mid * s_x, -j_mid * s_y, -k_mid * s_z],
                        dtype=torch.float32)
    
    return M, b
