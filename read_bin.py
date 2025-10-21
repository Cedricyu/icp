from pathlib import Path
import pycolmap
import numpy as np

from pathlib import Path
import numpy as np
import read_write_model as py


def parse_colmap(sparse_dir: str | Path):
    """
    使用 read_write_model 解析 COLMAP 重建資料，回傳：
    - images_pairs: List of list of [ [x, y], point3D_id ]
    - image_list: List of image names
    - point_list: Dict[point3D_id] = [X, Y, Z]
    - pose_list: Dict[image_name] = (R, t) ← camera pose
    """
    sparse_dir = Path(sparse_dir)
    images = py.read_images_binary(sparse_dir / "images.bin")
    points3D = py.read_points3D_binary(sparse_dir / "points3D.bin")
    cameras = py.read_cameras_binary(sparse_dir / "cameras.bin")

    images_pairs = {}
    pose_list = {}
    point_list = {pid: list(pt.xyz) for pid, pt in points3D.items()}
    intrinsics = {}

    for img_id, image in images.items():
        R = py.qvec2rotmat(image.qvec)
        t = np.array(image.tvec)
        pose_list[image.name] = [R, t]

        pairs = []
        for xy, pid in zip(image.xys, image.point3D_ids):
            if pid == -1:
                continue
            pairs.append([list(xy), pid])

        images_pairs[image.name] = pairs

        cam = cameras[image.camera_id]
        print("Camera:", cam.params)
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
            K = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,   1]
            ])
            intrinsics[image.name] = K
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1]
            ])
            intrinsics[image.name] = K
        else:
            print(f"[警告] 尚未支援相機模型：{cam.model}")

    return images_pairs, point_list, pose_list, intrinsics



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse', required=True, help='COLMAP sparse folder')
    args = parser.parse_args()

    pairs, image_list, point_list, _, _ = parse_colmap(args.sparse)
    # print("pairs:", pairs)
    for i, (img, p) in enumerate(zip(image_list, pairs)):
        print(f"[{i:03d}] {img}: {len(p)} matches")
    # print("image_list:", image_list)
    # print("point_list:", point_list)
