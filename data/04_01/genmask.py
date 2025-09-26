import os
import cv2
import numpy as np
import read_write_model as rw  # COLMAP 官方的 Python 讀寫腳本

sparse_path = "sparse"
image_dir = "images"
mask_dir = "masks"
os.makedirs(mask_dir, exist_ok=True)

cameras, images, points3D = rw.read_model(sparse_path, ext=".bin")

for image_id, im in images.items():
    # 讀影像取得尺寸
    img = cv2.imread(os.path.join(image_dir, im.name))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 從 images.bin 取出所有 keypoint
    for (x, y), pid in zip(im.xys, im.point3D_ids):
        if pid == -1:  # 沒有對應到 3D 點 → 跳過
            continue
        u, v = int(round(x)), int(round(y))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(mask, (u, v), radius=1, color=255, thickness=-1)

    out_path = os.path.join(mask_dir, f"{im.name}")
    cv2.imwrite(out_path, mask)
    print("Saved:", out_path)
