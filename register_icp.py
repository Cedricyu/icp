import open3d as o3d
import numpy as np
import argparse
import os


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def compute_fpfh(pcd_down, voxel_size):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))


def run_registration(source, target, voxel_size):
    source_down = preprocess(source, voxel_size)
    target_down = preprocess(target, voxel_size)

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
    )

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 1.0,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return result_icp


def save_result(i, transformation, source, target, rmse):
    transformed = source.transform(transformation.copy())
    combined = transformed + target

    o3d.io.write_point_cloud(f"outputs/aligned_{i:03d}.ply", combined)
    np.savetxt(f"outputs/transform_{i:03d}.txt", transformation)
    print(f"[Attempt {i:02d}] RMSE: {rmse:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    args = parser.parse_args()

    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)

    best_rmse = float('inf')
    best_result = None
    best_T = None

    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)

    for i in range(args.attempts):
        source = o3d.io.read_point_cloud(args.source)
        target = o3d.io.read_point_cloud(args.target)

        result = run_registration(source, target, args.voxel_size)
        rmse = result.inlier_rmse

        # 儲存結果
        save_result(i, result.transformation, o3d.io.read_point_cloud(args.source),
                                        o3d.io.read_point_cloud(args.target), rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_result = o3d.io.read_point_cloud(args.source)
            best_result.transform(result.transformation.copy())
            best_T = result.transformation.copy()

    # 上色顯示：source 為紅色，target 為綠色
    if best_result is not None:
        o3d.io.write_point_cloud("outputs/aligned_best.ply", best_result + target_raw)
        np.savetxt("outputs/transform_best.txt", best_T)
        print("Best result saved to aligned_best.ply with RMSE:", best_rmse)
        best_result.paint_uniform_color([1, 0, 0])   # 紅色
        target_raw.paint_uniform_color([0, 1, 0])    # 綠色
        o3d.visualization.draw_geometries([best_result, target_raw], window_name="ICP Best Alignment")
    else:
        print("No successful registration found — all attempts failed.")


if __name__ == '__main__':
    main()
