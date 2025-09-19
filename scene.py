import cv2
import numpy as np
from glob import glob
from sklearn.cluster import AgglomerativeClustering
import os
import shutil
import numpy as np

image_paths = sorted(glob("data/scene/*.png"))
features = []

orb = cv2.ORB_create()
for p in image_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)
    if des is None: 
        des = np.zeros((1,32))
    features.append(des.mean(axis=0))  # 用平均特徵當代表

features = np.array(features)

# 分成 20 組 (大約每組 20 張)
clustering = AgglomerativeClustering(n_clusters=20).fit(features)
labels = clustering.labels_
output_root = "scenes_output"
os.makedirs(output_root, exist_ok=True)

num_groups = max(labels) + 1  # 總共分幾組

for i in range(num_groups):
    # 找到屬於這組的 index
    group_idx = np.where(labels == i)[0]
    group_dir = os.path.join(output_root, f"scene_{i+1}/images")
    os.makedirs(group_dir, exist_ok=True)

    print(f"Scene {i+1}: {len(group_idx)} images")
    for idx in group_idx:
        src = image_paths[idx]
        dst = os.path.join(group_dir, os.path.basename(src))
        shutil.copy(src, dst)