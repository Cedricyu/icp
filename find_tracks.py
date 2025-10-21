
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree

def find_tracks(origin_features, projected_features,
                         kp1, des1, kp2, des2, indices,
                         img2,
                         search_radius=20, ratio=0.8,
                         visualize=True, max_draw=1000):
    kp1_pts = np.array([kp.pt for kp in kp1], dtype=np.float32)
    kp2_pts = np.array([kp.pt for kp in kp2], dtype=np.float32)

    tree1 = KDTree(kp1_pts)
    tree2 = KDTree(kp2_pts)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=32))

    refined_matches = []
    matched_indices = []

    for orig_pt, proj_pt, indice in zip(origin_features, projected_features, indices):
        # 找 origin 最近的 kp1
        _, idx1 = tree1.query(orig_pt)
        desc1 = des1[idx1].reshape(1, -1)

        # 找投影點附近的候選 kp2
        nearby_idx = tree2.query_ball_point(proj_pt, r=search_radius)
        if len(nearby_idx) == 0:
            refined_matches.append([np.nan, np.nan])
            matched_indices.append(indice)
            continue

        cand_desc = des2[nearby_idx].astype(np.float32)

        # --- 🔧 修正重點：候選太少時不跑 ratio test ---
        if len(nearby_idx) == 1:
            refined_matches.append(kp2[nearby_idx[0]].pt)
            matched_indices.append(indice)
            continue

        matches = flann.knnMatch(desc1.astype(np.float32), cand_desc, k=2)
        if len(matches[0]) < 2:
            refined_matches.append([np.nan, np.nan])
            matched_indices.append(indice)
            continue

        m, n = matches[0]
        if m.distance < ratio * n.distance:
            best_idx = nearby_idx[m.trainIdx]
            refined_matches.append(kp2[best_idx].pt)
        else:
            refined_matches.append([np.nan, np.nan])
        matched_indices.append(indice)

    refined_matches = np.array(refined_matches, dtype=np.float32)

    # ---------------- 視覺化 ----------------
    if visualize and len(refined_matches) > 0:
        img_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        valid_mask = ~np.isnan(refined_matches).any(axis=1)

        draw_indices = np.where(valid_mask)[0][:max_draw]
        for i in draw_indices:
            u, v = map(int, projected_features[i])
            mu, mv = map(int, refined_matches[i])
            cv2.circle(img_vis, (u, v), 2, (0, 255, 0), -1)
            cv2.circle(img_vis, (mu, mv), 2, (0, 0, 255), -1)
            cv2.line(img_vis, (u, v), (mu, mv), (255, 0, 0), 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Refined matches (valid: {valid_mask.sum()}/{len(valid_mask)})")
        plt.show()

    return refined_matches, matched_indices


def find_projections(image_list, pts3d, extrinsic, intrinsics, visualize=False):
    """
    將 3D 點投影到所有影像，輸出 dict
    projected_pts[image_name] = (P, 2) numpy array
    """
    projected_pts = {}

    for idx, name in enumerate(image_list):
        img_path = f"output/images/{name}"
        curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            raise FileNotFoundError(f"找不到圖片 {img_path}")

        pts_proj, valid_mask = project_and_visualize_points(
            name=name,
            target_frame_idx=idx,
            extrinsic=extrinsic,
            intrinsics=intrinsics,
            pts3d=pts3d,
            image_shape=curr_img.shape,
            visualize=visualize
        )

        projected_pts[name] = pts_proj
    return projected_pts

def project_and_visualize_points(
    name,                # 圖片名稱（不含路徑）
    target_frame_idx,       # 關鍵幀 index，用來取 intrinsics
    extrinsic,        # 所有相機姿態 dict：idx → (R, t)
    intrinsics,          # 相機內參 dict：idx → K
    pts3d,               # 3D 點 list 或 np.array
    image_shape,         # 圖像尺寸 (h, w)，用於邊界檢查
    img_root="output/images",  # 圖像資料夾路徑
    color_image=None,
    visualize=True       # 是否顯示圖像
):
    h, w = image_shape
    img_path = f"{img_root}/{name}"
    
    if color_image is not None:
        curr_img = color_image
    else:
        curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

    R, t = extrinsic[name]
    K = intrinsics[name]

    pts3d_np = np.asarray(pts3d)
    pts_cam = (R @ pts3d_np.T + t.reshape(3, 1)).T  # shape (N, 3)

    pts_proj = (K @ pts_cam.T).T  # shape (N, 3)
    pts_proj /= pts_proj[:, 2:3]  # normalize by z

    point_colors = []
    if color_image is not None:
        vis_img = curr_img.copy()
        for pt2d in pts_proj[:, :2]:
            x, y = int(round(pt2d[0])), int(round(pt2d[1]))
            if 0 <= x < w and 0 <= y < h:
                color = curr_img[y, x]  # shape: (3,) in BGR
                point_colors.append(color[::-1])  # Convert BGR to RGB
                if visualize:
                    cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
            else:
                point_colors.append(np.array([0, 0, 0]))  # fallback color if out of bounds

    if visualize:
        vis_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
        for x, y in pts_proj[:, :2].astype(int):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)

        plt.figure(figsize=(10, 6))
        plt.imshow(vis_img[..., ::-1])  # BGR to RGB
        plt.title(f"Projection onto {name}")
        plt.axis(False)
        plt.show()

    return pts_proj, np.array(point_colors)  # 可選回傳