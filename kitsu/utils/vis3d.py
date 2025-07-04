from pathlib import Path

import numpy as np
import torch
import torch as th
from skimage import measure
import matplotlib.pyplot as plt

# from pytorch3d.ops import cubify, sample_points_from_meshes
# from pytorch3d.structures import Meshes


def _get_fixed_random_color(index: int, colormap_name: str = "tab20") -> tuple:
    """
    Get a fixed but random color for a given index from a Matplotlib colormap.

    Parameters:
        index (int): The index to determine the color.
        colormap_name (str): The name of the Matplotlib colormap to use (default is 'tab20').

    Returns:
        list: An (R, G, B) tuple representing the color.
    """
    colormap = plt.get_cmap(colormap_name)
    # Normalize the index to ensure it fits within the colormap range
    normalized_index = index % colormap.N
    color = colormap(normalized_index)

    # Convert RGBA to RGB
    return list(color[:3])


def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    """
    Run marching cubes from PSR grid
    from Shape as Points
    """
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1]  # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size > 1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis=0)
        faces = np.stack(faces, axis=0)
        normals = np.stack(normals, axis=0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s - 1)  # scale to range [0, 1]
    else:
        verts = verts / s  # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals


def make_pointclouds_grid(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy or Tensor
    - return:
        - pts: N (3 or 6)
    """
    if isinstance(pts[0], torch.Tensor):
        return _make_pointclouds_grid_torch(pts, min_v, max_v, padding, nrows)
    elif isinstance(pts[0], np.ndarray):
        return _make_pointclouds_grid_numpy(pts, min_v, max_v, padding, nrows)
    else:
        raise TypeError


def _make_pointclouds_grid_numpy(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), numpy
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = np.concatenate(out_pts, 0)  # N (3 or 6)
    return pts


@torch.no_grad()
def _make_pointclouds_grid_torch(pts, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - pts: list of (n (3 or 6)), Tensor
    - return:
        - pts: N (3 or 6)
    """
    dist = max_v - min_v
    out_pts = []
    for i in range(len(pts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = pts[i].new_tensor([[off_x, off_y, *((0,) * (pts[0].shape[-1] - 2))]])
        out_pts.append(pts[i] + offset)
    pts = torch.cat(out_pts, 0)  # N (3 or 6)
    return pts


def make_meshes_grid(verts, faces, min_v, max_v, padding=1, nrows=8):
    """
    - input:
        - verts: list of (n (3 ~)), numpy
        - faces: list of n 3, numpy, int
    - return:
        - verts: n (3 ~)
        - faces: n 3
    """
    assert len(verts) == len(faces)

    dist = max_v - min_v
    face_offset = 0
    out_verts, out_faces = [], []
    for i in range(len(verts)):
        pos_x, pos_y = i % nrows, i // nrows
        off_x = pos_x * (dist + padding)
        off_y = pos_y * (dist + padding)
        offset = np.array([[off_x, off_y, *((0,) * (verts[0].shape[-1] - 2))]])
        out_verts.append(verts[i] + offset)
        out_faces.append(faces[i] + face_offset)
        face_offset += verts[i].shape[0]
    verts = np.concatenate(out_verts)  # N (3 or 6)
    faces = np.concatenate(out_faces)  # N 3
    return verts, faces


def random_color(verts):
    """
    - input:
        - verts: n 3
    """
    color = np.random.random(1, 3)
    color = np.repeat(color, verts.shape[0], 0)  # n 3
    return np.concatenate([verts, color], -1)  # n 6


def sdfs_to_meshes(psrs, safe=False):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - meshes
    """
    from pytorch3d.structures import Meshes

    mvs, mfs, mns = [], [], []
    for psr in psrs:
        if safe:
            try:
                mv, mf, mn = mc_from_psr(psr, pytorchify=True)
            except:
                mv = psrs.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                mf = psrs.new_tensor([[0, 1, 2]], dtype=torch.long)
                mn = psrs.new_tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        else:
            mv, mf, mn = mc_from_psr(psr, pytorchify=True)
        mvs.append(mv)
        mfs.append(mf)
        mns.append(mn)

    mesh = Meshes(mvs, mfs, verts_normals=mns)
    return mesh


def sdfs_to_meshes_np(psrs, safe=False, rescale_verts=False):
    """
    - input:
        - psrs: b 1 r r r
    - return:
        - verts: list of (n 3)
        - faces: list of (m 3)
    """
    mesh = sdfs_to_meshes(psrs, safe=safe)
    vs1, fs1 = mesh.verts_list(), mesh.faces_list()
    vs2, fs2 = [], []
    for i in range(len(vs1)):
        v = (vs1[i] * 2 - 1) if rescale_verts else vs1[i]
        vs2.append(v.cpu().numpy())
        fs2.append(fs1[i].cpu().numpy())
    return vs2, fs2


def sdf_to_point(sdf, n_points, safe=False):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes

    if safe:
        try:
            mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
            mesh = Meshes([mv], [mf], verts_normals=[mn])
            pts = sample_points_from_meshes(mesh, n_points)
        except RuntimeError:
            pts = sdf.new_zeros(1, n_points, 3)
    else:
        mv, mf, mn = mc_from_psr(sdf, pytorchify=True)
        mesh = Meshes([mv], [mf], verts_normals=[mn])
        pts = sample_points_from_meshes(mesh, n_points)

    return pts[0]


def sdfs_to_points(sdfs, n_points, safe=False):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point(sdf, n_points, safe=safe) for sdf in sdfs])


def sdf_to_point_fast(sdf, n_points):
    """
    - input:
        - sdf: 1 r r r
    - return:
        - point: n_points 3
    """
    from pytorch3d.ops import cubify, sample_points_from_meshes

    mesh = cubify(-sdf, 0)
    pts = sample_points_from_meshes(mesh, n_points)
    return pts[0]


def sdfs_to_points_fast(sdfs, n_points):
    """
    - input:
        - sdfs: b 1 r r r
    - return:
        - points: b n_points 3
    """
    return torch.stack([sdf_to_point_fast(sdf, n_points) for sdf in sdfs])


def save_sdf_as_mesh(path, sdf, safe=False):
    """
    - input:
        - sdf: 1 r r r
    """
    import point_cloud_utils as pcu

    verts, faces = sdfs_to_meshes_np(sdf[None], safe=safe)
    pcu.save_mesh_vf(str(path), verts[0], faces[0])


def udf2mesh(udf, grad, b_max, b_min, resolution):
    """
    - udf: r r r, numpy
    - grad: r r r 3, numpy
    """
    import mcubes

    v_all = []
    f_all = []
    threshold = 0.005  # accelerate extraction
    v_num = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                ndf_loc = udf[i : i + 2]
                ndf_loc = ndf_loc[:, j : j + 2, :]
                ndf_loc = ndf_loc[:, :, k : k + 2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i : i + 2]
                grad_loc = grad_loc[:, j : j + 2, :]
                grad_loc = grad_loc[:, :, k : k + 2]

                res = np.ones((2, 2, 2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min() < 0:
                    vertices, triangles = mcubes.marching_cubes(res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:, 0] += i  # / resolution
                    vertices[:, 1] += j  # / resolution
                    vertices[:, 2] += k  # / resolution
                    triangles += v_num
                    # vertices =
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    f_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    f_all = np.concatenate(f_all)
    # Create mesh
    v_all = v_all / (resolution - 1.0) * (b_max - b_min)[None, :] + b_min[None, :]

    return v_all, f_all


def save_point_cloud(filename, vertices, colors=None, normals=None):
    """
    Save a point cloud to a PLY file (ASCII) without using Open3D.

    Parameters:
    - filename: str, path to output .ply file.
    - vertices: (N, 3) numpy or torch array of xyz coordinates.
    - colors: (N, 3) numpy or torch array of uint8 RGB values (0–255).
    - normals: (N, 3) numpy or torch array of normal vectors.
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if normals is not None and isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().numpy()

    num_points = vertices.shape[0]
    use_color = colors is not None
    use_normal = normals is not None
    assert vertices.ndim == 2
    if use_color:
        assert colors.ndim == 2
    if use_normal:
        assert normals.ndim == 2

    with open(filename, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if use_normal:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if use_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        # Write each point
        for i in range(num_points):
            v = vertices[i]
            line = f"{v[0]} {v[1]} {v[2]}"
            if use_normal:
                n = normals[i]
                line += f" {n[0]} {n[1]} {n[2]}"
            if use_color:
                c = colors[i].astype(np.uint8)
                line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")


def save_point_clouds(filename, *pcds):
    import open3d as o3d

    cmap = plt.get_cmap("tab20")

    out_pcds = []
    out_colors = []

    for i, pcd in enumerate(pcds):
        if isinstance(pcd, torch.Tensor):
            pcd = pcd.detach().cpu().numpy()
        assert pcd.ndim == 2 and pcd.shape[1] == 3, pcd.shape

        color = cmap(i % cmap.N)[:3]
        color = np.array(color, dtype=np.float32)
        color = np.repeat(color[None], pcd.shape[0], 0)  # n 3

        out_pcds.append(pcd)
        out_colors.append(color)

    pcd = np.concatenate(out_pcds)
    color = np.concatenate(out_colors)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pcd)
    out.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(str(filename), out)


def save_mesh(filename, vertices, faces, colors=None, normals=None):
    """
    Save a mesh to a PLY file using open3d.

    Parameters:
    - filename: Name of the PLY file to save to.
    - vertices: Nx3 numpy array or torch.Tensor of vertex positions.
    - faces: Mx3 numpy array or torch.Tensor of triangular indices.
    - colors (optional): Nx3 numpy array or torch.Tensor of vertex colors in uint8 format (range [0, 255]).
                         If None, only vertices and faces are saved.
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
        if colors.dtype == np.int8:
            colors = colors.astype(np.float32) / 255.0
    if normals is not None and isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().numpy()

    txt = []
    for i in range(vertices.shape[0]):
        if colors is not None:
            txt.append(f"v {vertices[i,0]} {vertices[i,1]} {vertices[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}")
        else:
            txt.append(f"v {vertices[i,0]} {vertices[i,1]} {vertices[i,2]}")

    if normals is not None:
        for n in normals:
            txt.append(f"vn {n[0]} {n[1]} {n[2]}")

    for face in faces:
        f0, f1, f2 = face + 1
        if normals is not None:
            txt.append(f"f {f0}//{f0} {f1}//{f1} {f2}//{f2}")
        else:
            txt.append(f"f {f0} {f1} {f2}")

    with open(filename, "w") as f:
        f.write("\n".join(txt))


# def open_point_cloud(filename):
#     import open3d as o3d

#     point_cloud = o3d.io.read_point_cloud(filename)
#     points = np.asarray(point_cloud.points)
#     tensor = th.from_numpy(points).float()
#     return tensor


def open_point_cloud(filename):
    """
    Read a point cloud from an ASCII .ply file without using Open3D.

    Returns:
    - torch.Tensor of shape (N, 3) containing xyz coordinates.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse header
    header_ended = False
    vertex_count = 0
    header_lines = []
    for i, line in enumerate(lines):
        header_lines.append(line.strip())
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_ended = True
            start_idx = i + 1
            break

    assert header_ended, "Invalid PLY file: missing end_header"

    # Read vertex data
    vertices = []
    for line in lines[start_idx : start_idx + vertex_count]:
        tokens = line.strip().split()
        xyz = list(map(float, tokens[:3]))
        vertices.append(xyz)

    vertices = np.array(vertices, dtype=np.float32)
    return torch.from_numpy(vertices)
