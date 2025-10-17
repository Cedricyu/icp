import os
import argparse
import numpy as np
import read_write_model as rw  # 放在同一資料夾下


def transform_points3D(points3D, T):
    """Apply 4x4 transformation to all 3D points."""
    for pid, pt in points3D.items():
        xyz_h = np.hstack([pt.xyz, 1.0])  # [x, y, z, 1]
        xyz_new = T @ xyz_h
        points3D[pid] = rw.Point3D(
            id=pt.id,
            xyz=xyz_new[:3],
            rgb=pt.rgb,
            error=pt.error,
            image_ids=pt.image_ids,
            point2D_idxs=pt.point2D_idxs,
        )
    return points3D

def merge_models(src_dir, tgt_dir, src_T_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 讀變換矩陣 T (source → target)
    T = np.loadtxt(src_T_path)
    T = np.asarray(T, dtype=np.float64)

    # 將 3×4 或 4×3 正規化為 4×4 齊次；若已是 4×4 則保持
    if T.shape == (3, 4):
        T = np.vstack([T, np.array([0, 0, 0, 1.0])])
    elif T.shape == (4, 3):
        T = np.hstack([T, np.array([[0], [0], [0], [1.0]])])
    elif T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4/3x4/4x3, got {T.shape}")

    # 讀取 source / target models
    cams_s, imgs_s, pts_s = rw.read_model(src_dir, ext=".bin")
    cams_t, imgs_t, pts_t = rw.read_model(tgt_dir, ext=".bin")

    # 小工具：安全取得 offset（目標集合可能為空）
    def safe_max_plus_one(d):
        return (max(d.keys()) + 1) if len(d) > 0 else 1

    cam_id_offset = safe_max_plus_one(cams_t)
    img_id_offset = safe_max_plus_one(imgs_t)   # ✅ 修正：以 target 的最大 id + 1
    pt_id_offset  = safe_max_plus_one(pts_t)

    # 1) 改相機 ID（防止衝突）
    new_cams_s = {}
    cam_id_map = {}
    for cid, cam in cams_s.items():
        new_cid = cid + cam_id_offset
        cam_id_map[cid] = new_cid
        # 依你的 Camera 結構保留欄位
        new_cams_s[new_cid] = rw.Camera(
            id=new_cid,
            model=cam.model,
            width=cam.width,
            height=cam.height,
            params=cam.params,
            # 你的 rw 似乎支援折射模型，照原樣拷貝（若沒有就保留 None/[]）
            refrac_model=getattr(cam, "refrac_model", None),
            refrac_params=getattr(cam, "refrac_params", []),
        )

    # 2) 改 image ID / camera ID，並將相機位姿從 source-world 映射到 target-world
    new_imgs_s = {}
    for iid, img in imgs_s.items():
        new_iid = iid + img_id_offset

        # 世界→相機： x_cam = R * X_world + t
        R = rw.qvec2rotmat(img.qvec)          # world->cam
        t = img.tvec.reshape(3, 1)            # world->cam

        # 相機在世界中的位姿 (world←cam)： [Rwc | twc]
        Rwc = R.T
        twc = -R.T @ t

        T_c2 = np.eye(4, dtype=np.float64)
        T_c2[:3, :3] = Rwc
        T_c2[:3, 3]  = twc.reshape(3)

        # 將相機中心位姿從 source-world 轉到 target-world
        # X_target = T * X_source → 位姿也：T_c2_target = T @ T_c2
        T_new = T @ T_c2

        # 還原為 COLMAP 的 (R, t)（world→cam）
        Rwc_new = T_new[:3, :3]
        twc_new = T_new[:3, 3].reshape(3, 1)
        R_new = Rwc_new.T
        t_new = (-R_new @ twc_new).reshape(3)
        qvec_new = rw.rotmat2qvec(R_new)

        # 這張影像對應的 point3D_ids 也要加上新的偏移（-1 保留）
        if img.point3D_ids is None:
            new_point3D_ids = None
        else:
            new_point3D_ids = np.array(
                [(-1 if pid == -1 else pid + pt_id_offset) for pid in img.point3D_ids],
                dtype=np.int64
            )

        new_imgs_s[new_iid] = rw.Image(
            id=new_iid,
            qvec=qvec_new,
            tvec=t_new,
            camera_id=cam_id_map[img.camera_id],
            name=img.name,
            xys=img.xys,
            point3D_ids=new_point3D_ids
        )

    # 3) 改 point3D ID 並套用 T（source-world → target-world）
    new_pts_s = {}
    for pid, pt in pts_s.items():
        new_pid = pid + pt_id_offset
        xyz_h = np.hstack([pt.xyz, 1.0])           # 齊次
        xyz_new_h = T @ xyz_h
        xyz_new = xyz_new_h[:3]

        # 這個點被哪些影像觀測到：image_ids 也需映射到新 image id
        if pt.image_ids is None:
            new_image_ids = None
        else:
            new_image_ids = [iid + img_id_offset for iid in pt.image_ids]

        new_pts_s[new_pid] = rw.Point3D(
            id=new_pid,
            xyz=xyz_new,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=new_image_ids,
            point2D_idxs=pt.point2D_idxs,
        )

    # 4) 合併
    merged_cams = {**cams_t, **new_cams_s}
    merged_imgs = {**imgs_t, **new_imgs_s}
    merged_pts  = {**pts_t,  **new_pts_s}

    # 5) 輸出
    rw.write_model(merged_cams, merged_imgs, merged_pts, path=output_dir, ext=".bin")
    print(f"[INFO] ✅ Merged model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_sparse', required=True, help='source sparse folder (要轉換)')
    parser.add_argument('--target_sparse', required=True, help='target sparse folder (基準)')
    parser.add_argument('--transform', required=True, help='source → target 的 transform_best.txt')
    parser.add_argument('--output', required=True, help='輸出合併後的 sparse 資料夾')
    args = parser.parse_args()

    merge_models(args.source_sparse, args.target_sparse, args.transform, args.output)
