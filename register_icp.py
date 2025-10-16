import os
import copy
import numpy as np
import open3d as o3d
from probreg import cpd
import argparse


def estimate_density_based_voxel(pcd, base_factor=3.0):
    """根據平均鄰近距離估計合適的 voxel size"""
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    pts = np.asarray(pcd.points)
    for i in range(0, len(pts), max(1, len(pts)//500)):  # 採樣500個點加速
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 6)
        if len(idx) > 1:
            local_dists = np.linalg.norm(pts[idx[1:]] - pts[i], axis=1)
            dists.append(np.mean(local_dists))
    mean_nn = np.mean(dists)
    voxel_size = mean_nn * base_factor
    print(f"[Auto voxel] mean_nn = {mean_nn:.5f}, voxel_size = {voxel_size:.5f}")
    return voxel_size


def preprocess(pcd, voxel_size=None):
    """下採樣 + 統計濾波去除噪點"""
    print(f"Original points: {len(pcd.points)}")

    # === Step 1: 統計濾波 ===
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After denoise: {len(pcd.points)} points")

    # === Step 2: 自動估計 voxel ===
    if voxel_size is None:
        voxel_size = estimate_density_based_voxel(pcd)

    # === Step 3: 下採樣 ===
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.remove_non_finite_points()
    print(f"After voxel downsample ({voxel_size:.5f}): {len(pcd.points)} points\n")

    return pcd, voxel_size


def run_registration(source, target):
    """使用 CPD 進行剛體註冊"""
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name='rigid', w=0.1, maxiter=100)
    return tf_param


def save_result(i, transformation, source, target, rmse):
    """儲存轉換後的結果與 RMSE"""
    transformed = copy.deepcopy(source)
    transformed.transform(transformation)
    combined = transformed + target
    os.makedirs("outputs", exist_ok=True)
    o3d.io.write_point_cloud(f"outputs/aligned_{i:03d}.ply", combined)
    np.savetxt(f"outputs/transform_{i:03d}.txt", transformation)
    print(f"[Attempt {i:02d}] RMSE: {rmse:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--voxel_size', type=float, default=None, help="若不指定會自動根據密度估計")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # === 讀入點雲 ===
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)

    print("Source:", source_raw)
    print("Target:", target_raw)
    print("Source bounds:", source_raw.get_min_bound(), source_raw.get_max_bound())
    print("Target bounds:", target_raw.get_min_bound(), target_raw.get_max_bound())

    # === 前處理：去噪 + 下採樣 ===
    source, voxel_s = preprocess(source_raw, args.voxel_size)
    target, voxel_t = preprocess(target_raw, args.voxel_size)
    voxel_size = (voxel_s + voxel_t) / 2

    # === 多次嘗試 ===
    best_rmse = float('inf')
    best_trans = None
    best_aligned = None

    for i in range(args.attempts):
        print(f"\n=== Attempt {i + 1}/{args.attempts} ===")

        tf_param = run_registration(source, target)
        transformed = copy.deepcopy(source)
        transformed.points = tf_param.transform(transformed.points)

        # 計算 RMSE
        dists = np.asarray(target.compute_point_cloud_distance(transformed))
        rmse = np.sqrt(np.mean(dists ** 2))

        # 組成 4x4 變換矩陣
        T = np.eye(4)
        T[:3, :3] = tf_param.rot * tf_param.scale
        T[:3, 3] = tf_param.t.ravel()

        save_result(i, T, source, target, rmse)

        # 更新最佳
        if rmse < best_rmse:
            best_rmse = rmse
            best_trans = T.copy()
            best_aligned = transformed

    # === 儲存最佳結果 ===
    if best_aligned is not None:
        o3d.io.write_point_cloud("outputs/aligned_best.ply", best_aligned + target_raw)
        np.savetxt("outputs/transform_best.txt", best_trans)
        print(f"\n✅ Best alignment saved (RMSE = {best_rmse:.6f})")

        best_aligned.paint_uniform_color([1, 0, 0])
        target_raw.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([best_aligned, target_raw], window_name="CPD Best Alignment")
    else:
        print("❌ No successful registration found.")


if __name__ == "__main__":
    main()
