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

    # 讀取 source model（要轉換）
    cams_s, imgs_s, pts_s = rw.read_model(src_dir, ext=".bin")
    # 讀取 target model（保持不變）
    cams_t, imgs_t, pts_t = rw.read_model(tgt_dir, ext=".bin")

    # 改相機 ID（防止衝突）
    cam_id_offset = max(cams_t.keys()) + 1
    new_cams_s = {}
    cam_id_map = {}
    for cid, cam in cams_s.items():
        new_cid = cid + cam_id_offset
        cam_id_map[cid] = new_cid
        print("model :",cam.model)
        new_cams_s[new_cid] = rw.Camera(
            id=new_cid,
            model=cam.model,
            width=cam.width,
            height=cam.height,
            params=cam.params,
            refrac_model=None,
            refrac_params=[],
        )

    # 改 image ID 和 camera ID，並套用 T 到 pose
    img_id_offset = max(imgs_s.keys())-1
    new_imgs_s = {}
    for iid, img in imgs_s.items():
        new_iid = iid + img_id_offset
        print(new_iid, img.name)
        # 姿態轉換：R, t → pose matrix → T @ pose → R', t'
        R = rw.qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)

        Rwc = R.T
        twc = -R.T @ t
        T_c2 = np.eye(4)
        T_c2[:3, :3] = Rwc
        T_c2[:3, 3:] = twc

        T_new = T @ T_c2  
        R_new = T_new[:3, :3].T
        t_new = -R_new @ T_new[:3, 3]
        qvec_new = rw.rotmat2qvec(R_new)
        new_imgs_s[new_iid] = rw.Image(
            id=new_iid,
            qvec=qvec_new,
            tvec=t_new,
            camera_id=cam_id_map[img.camera_id],
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )

    # 改 point3D ID 並套用 T
    pt_id_offset = max(pts_t.keys()) + 1
    new_pts_s = {}
    for pid, pt in pts_s.items():
        new_pid = pid + pt_id_offset
        xyz_h = np.hstack([pt.xyz, 1.0])  # 齊次座標
        xyz_new = T @ xyz_h
        new_pts_s[new_pid] = rw.Point3D(
            id=new_pid,
            xyz=xyz_new[:3],
            rgb=pt.rgb,
            error=pt.error,
            image_ids=[iid + img_id_offset for iid in pt.image_ids],
            point2D_idxs=pt.point2D_idxs,
        )

    # 合併所有資訊
    merged_cams = {**cams_t, **new_cams_s}
    merged_imgs = {**imgs_t, **new_imgs_s}
    merged_pts = {**pts_t, **new_pts_s}

    # 輸出為 .txt 格式（可轉為 .bin）
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
