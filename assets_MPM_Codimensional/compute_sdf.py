import numpy as np
import open3d as o3d

def compute_sdf_open3d(obj_filename, grid_resolution=64):
    """
    使用 Open3D 计算给定 OBJ 网格在规则网格上的 SDF。
    """
    print("加载 OBJ 文件...")
    mesh = o3d.io.read_triangle_mesh(obj_filename)
    if not mesh.is_edge_manifold() or mesh.is_self_intersecting():
        print("警告: 网格可能不是闭合的，这可能导致 SDF 计算出现问题。")

    print("获取网格边界...")
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()

    print("创建规则网格点...")
    xs = np.linspace(min_bound[0], max_bound[0], grid_resolution)
    ys = np.linspace(min_bound[1], max_bound[1], grid_resolution)
    zs = np.linspace(min_bound[2], max_bound[2], grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    print("创建 KD 树...")
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    print("计算签名距离...")
    signed_distances = np.zeros(grid_points.shape[0], dtype=np.float32)
    for i, point in enumerate(grid_points):
        [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
        signed_distances[i] = np.sqrt(dist[0])

        # 简单判断点是否在网格内部（基于射线投射奇偶法）
        ray = np.array([0, 0, 1])
        intersections = mesh.ray.intersect_ray(point, ray)
        count = len(intersections)
        if count % 2 == 1:
            signed_distances[i] = -signed_distances[i]

        if i % 10000 == 0 and i > 0:
            print(f"已处理 {i} / {grid_points.shape[0]} 点")

    print("重新塑形 SDF 数组...")
    sdf_np = signed_distances.reshape((grid_resolution, grid_resolution, grid_resolution)).astype(np.float32)

    print("SDF 计算完成。")
    return sdf_np

if __name__ == '__main__':
    obj_path = "assets_geometry/pighead.obj"  # 替换为你的 OBJ 文件路径
    try:
        sdf = compute_sdf_open3d(obj_path, grid_resolution=32)  # 尝试较低的分辨率
        np.save("assets_geometry/pighead_sdf_open3d.npy", sdf)
        print("SDF 计算完成并保存为 'pighead_sdf_open3d.npy'")
    except Exception as e:
        print(f"计算 SDF 过程中出错: {e}")
