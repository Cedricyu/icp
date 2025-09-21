import cv2
import numpy as np
from glob import glob
import networkx as nx
import matplotlib.pyplot as plt
import os

# === 參數設定 ===
image_dir = "data/scene"
min_inliers = 50
min_overlap = 0.05

# === 讀取圖片並抽 SIFT 特徵 ===
image_paths = sorted(glob(f"{image_dir}/*.png"))
sift = cv2.SIFT_create()

keypoints, descriptors = [], []
for p in image_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# === 建立比對關係圖 ===
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
G = nx.Graph()

for i in range(len(image_paths)):
    G.add_node(i, filename=os.path.basename(image_paths[i]))

for i in range(len(image_paths)):
    for j in range(i+1, len(image_paths)):
        if descriptors[i] is None or descriptors[j] is None:
            continue

        matches = bf.knnMatch(descriptors[i], descriptors[j], k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 0:
            pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good])
            pts2 = np.float32([keypoints[j][m.trainIdx].pt for m in good])

            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 3.0)
            if F is not None and mask is not None:
                inliers = int(mask.sum())
                overlap = inliers / min(len(keypoints[i]), len(keypoints[j]))

                if inliers >= min_inliers and overlap >= min_overlap:
                    G.add_edge(i, j, weight=inliers)

# === 視覺化 ===
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, weight="weight", seed=42)

# 分強邊 / 弱邊
edges = G.edges(data=True)
strong_edges = [(u, v) for u, v, d in edges if d["weight"] >= 200]
weak_edges   = [(u, v) for u, v, d in edges if 50 <= d["weight"] < 200]

# 節點
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", alpha=0.9)

# 強邊（粗藍）
nx.draw_networkx_edges(G, pos, edgelist=strong_edges,
    width=2.5, edge_color="navy", alpha=0.8)

# 弱邊（細灰）
nx.draw_networkx_edges(G, pos, edgelist=weak_edges,
    width=1, edge_color="gray", alpha=0.4)

# 節點標籤（用 index）
labels = {i: str(i) for i in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title("Image Matching Graph (nodes=images, edges=inliers)", fontsize=14)
plt.axis("off")
plt.show()

# === 輸出 index 對應檔名 ===
print("\nIndex to filename mapping:")
for i, path in enumerate(image_paths):
    print(f"{i}: {os.path.basename(path)}")


import pandas as pd

relations = []  # 存邊的資料

# 當你 add_edge 的時候，把數據記下來
if inliers >= min_inliers and overlap >= min_overlap:
    G.add_edge(i, j, weight=inliers)
    relations.append({
        "img1": os.path.basename(image_paths[i]),
        "img2": os.path.basename(image_paths[j]),
        "inliers": inliers,
        "overlap": overlap
    })

# ...畫圖完之後
df = pd.DataFrame(relations)
df.to_csv("match_relations.csv", index=False)
print("➡ 已輸出 match_relations.csv 和 match_graph.png")
