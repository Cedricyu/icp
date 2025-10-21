
import pycolmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from read_bin import parse_colmap
from find_tracks import find_tracks, find_projections
import cv2
from scipy.spatial import KDTree
import os

print(pycolmap.__version__)

def draw_matches_side_by_side(img1, img2, pts1, pts2, max_draw=200):
    """
    在兩張圖之間畫出匹配:
    img1, img2: 灰階或彩色圖片
    pts1, pts2: Nx2 numpy array, 分別是 img1 / img2 上的點
    """
    # 確保彩色
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 拼接圖片 (左右)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 畫點和線
    for (x1, y1), (x2, y2) in zip(pts1[:max_draw], pts2[:max_draw]):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2) + w1, int(y2)), 3, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, 1)

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Matches shown: {min(len(pts1), max_draw)}")
    plt.show()

    return canvas

# %%
def filter_origin_by_sift(origin_features, kp1, max_dist=5):
    """
    origin_features: (N,2) 投影點
    kp1: SIFT keypoints (list of cv2.KeyPoint)
    max_dist: 最大距離閾值 (像素)，太遠就不算匹配

    回傳:
      filtered_origins: 有 SIFT 支撐的投影點 (array)
      pairs: [(origin_pt, sift_pt, pixel_error, origin_idx), ...]
      indices: 被保留的 origin 在 origin_features 中的 index
    """
    origin_features = np.array(origin_features)
    filtered_origins = []
    pairs = []
    indices = []

    for kp in kp1:
        sift_pt = np.array(kp.pt)
        # 找最近的 origin
        dists = np.linalg.norm(origin_features - sift_pt, axis=1)
        idx = np.argmin(dists)
        nearest_origin = origin_features[idx]
        nearest_dist = dists[idx]

        if nearest_dist < max_dist:
            filtered_origins.append(nearest_origin)
            indices.append(idx)
            pairs.append((nearest_origin, tuple(sift_pt), nearest_dist, idx))

    return np.array(filtered_origins), pairs, np.array(indices)

def drawkeypoints(img ,kp):
    print(f"偵測到 {len(kp)} 個 keypoints")

    # 畫在灰階圖上
    img_vis = cv2.drawKeypoints(
        img, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,  # 會顯示大小和方向
        color=(0, 255, 0)
    )

    # 顯示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("SIFT Keypoints on img")
    plt.show()

def visualize_track_matrix(tracks, image_list):
    """
    tracks: (N, P, 2)
    image_list: list of image names
    """
    # 建立一個有效性矩陣 (True = 有值, False = NaN)
    valid_mask = ~np.isnan(tracks[:, :, 0])
    coverage = np.sum(valid_mask) / valid_mask.size * 100

    plt.figure(figsize=(14, 6))
    plt.imshow(valid_mask, cmap='Greens', interpolation='nearest', aspect='auto')
    plt.title(f"Track Filling Progress — {coverage:.2f}% filled")
    plt.xlabel("3D Point ID")
    plt.ylabel("Image Index")
    plt.yticks(np.arange(len(image_list)), image_list, fontsize=8)
    plt.colorbar(label="Has Match (1=True, 0=False)")
    plt.tight_layout()
    plt.show()


def alignment(colmap_dir="output/sparse/0" ,source_dir="data/duck_bt/images", target_dir="data/duck_tp/images"):
    pairs, point_list, extrinsic, intrinsics = parse_colmap(colmap_dir)

    print(f"Total images: {len(pairs.keys())}")

    source_image_list = sorted(os.listdir(source_dir))
    target_image_list = sorted(os.listdir(target_dir))

    print("Source images:", source_image_list)
    print("Target images:", target_image_list)

    # print("pairs :", pairs)

    P = len(point_list)
    N = len(pairs.keys())
    image_list = list(pairs.keys())
    print(f"Total 3D points: {P}, Total images: {N}")
    # 初始化 tracks [N張圖片, P個3D點, (x, y)]
    tracks = np.full((N, P, 2), np.nan, dtype=np.float32)
    colors = np.full((P, 3), np.nan, dtype=np.float32)  # 每個 3D 點的顏色 (R,G,B)

    for i, image_name in enumerate(pairs.keys()):
        if image_name not in pairs:
            print(f"Warning: No pairs found for image {image_name}")
            continue
        # print(pairs[image_name])
        # pairs[image_name] = [(x, y, point3D_id), ...]
        for ((x, y)  ,pid) in pairs[image_name]:
            if pid < 0 or pid >= P:
                continue  # 避免超出範圍
            tracks[i, pid] = (x, y)

    # visualize_track_matrix(tracks, image_list)
    print("✅ Tracks filled.")


    sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.0005, sigma=1.0)

    sift_cache = {}
    for img_name in image_list:
        img = cv2.imread(f"output/images/{img_name}", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f"output/images_mask/{img_name}", cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, mask)
        sift_cache[img_name] = (kp, des)

    for key_frame_name in image_list:

        pair = pairs[key_frame_name]
        point_ids = [pid for (_, pid) in pair]
       
        pts3d = np.array([point_list[pid] for pid in point_ids if pid in point_list], dtype=np.float32)

        projected_pts = find_projections(image_list, pts3d, extrinsic, intrinsics, visualize=False)

        kp1, des1 = sift_cache[key_frame_name]
        origin_features = projected_pts[key_frame_name][:, :2]
        indices = point_ids  

        for i, image_name in enumerate(image_list):

            if (key_frame_name in source_image_list and image_name in source_image_list) or \
            (key_frame_name in target_image_list and image_name in target_image_list):
                continue
            
            img2_path = f"output/images/{image_name}"
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            kp2, des2 = sift_cache[image_name]
            projected_features = projected_pts[image_name][:, :2]

            refined_points, matched_indices = find_tracks(
                origin_features, projected_features,
                kp1, des1, kp2, des2, indices,
                img2, search_radius=5, visualize=False
            )

            valid_mask = ~np.isnan(refined_points[:, 0]) & ~np.isnan(refined_points[:, 1])
            if np.sum(valid_mask) < 8:
                print(f"⚠️ 有效對應點不足 ({np.sum(valid_mask)} < 8)，跳過 F 計算")
                continue

            F, mask = cv2.findFundamentalMat(
                origin_features[valid_mask],
                refined_points[valid_mask],
                cv2.FM_RANSAC,
                ransacReprojThreshold=2.0,
                confidence=0.99
            )

            if F is None or mask is None:
                print(f"⚠️ {image_name}: F estimation failed")
                continue

            inlier_mask = mask.ravel() == 1
            inlier_proj = origin_features[valid_mask][inlier_mask]
            inlier_match = refined_points[valid_mask][inlier_mask]
            inlier_track = np.array(matched_indices)[valid_mask][inlier_mask]


            new_count = 0
            for j, pid in enumerate(inlier_track):
                if np.isnan(tracks[i, pid-2]).all():
                    tracks[i, pid-2] = inlier_match[j]
                    new_count += 1
                    x, y = map(int, inlier_match[j])
                    if 0 <= x < img2.shape[1] and 0 <= y < img2.shape[0]:
                        bgr = img2[y, x] if len(img2.shape) == 3 else [img2[y, x]] * 3
                        colors[pid-2] = np.array(bgr[::-1], dtype=np.float32)

    key_frame_idx = 0
    key_frame_name = image_list[key_frame_idx]  # e.g., "0001.jpg"
    print(f"Processing keyframe: {key_frame_name}")
    # 讀 keyframe 的 RGB 圖片
    img_path = f"output/images/{key_frame_name}"
    key_img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    if key_img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    image_size = np.array(key_img.shape[:2][::-1])  # (W, H)
    # print(extrinsic)
    # for extrinsics in extrinsic.values():
    #     R, t = extrinsics
    #     print("R:", R)
    #     print("t:", t)
    extrinsics_array = np.stack([np.hstack((R, t.reshape(-1, 1))) for (R, t) in extrinsic.values()])
    # print(intrinsics.values())
    intrinsics_array = np.stack([intrinsic for intrinsic in  intrinsics.values()])
    print("image size :", image_size)
    print(len(intrinsics_array))

    sorted_items = sorted(point_list.items())  # [(key, value), ...]
    keys = np.array(sorted(point_list.keys()))
    all_points = np.array([point_list[k] for k in keys])
    print(len(all_points), tracks.shape)

    from batch_np_matrix_to_pycolmap import batch_np_matrix_to_pycolmap

    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        all_points,
        extrinsics_array,
        intrinsics_array,
        tracks,
        image_size,
        masks=None,
        max_reproj_error=25,
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
        image_list=image_list,
        points_rgb=colors,
    )

    if reconstruction is None:
        raise ValueError("No reconstruction can be built with BA")

    print("fininshed building pycolmap reconstruction")

    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    output_dir = "results/sparse/0"
    reconstruction.write(output_dir)
    print(f"Bundle adjustment result saved to {output_dir}")

def main():
    alignment()

if __name__ == "__main__":
    main()



