# %%
import torch

# 下載並載入 ViT-L/14 權重
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval().cuda()  # 若有 GPU

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# %%
import numpy as np

def extract_dino_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat = model(x)               # (1, 1024)
    v = feat[0].cpu().numpy()
    return v / np.linalg.norm(v)      # L2-normalize

# %%
import os
from tqdm import tqdm

image_dir = "data/scene"  # 你的影像資料夾
image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

embeddings = []
for p in tqdm(image_paths):
    embeddings.append(extract_dino_feature(p))
embeddings = np.vstack(embeddings)
print("Feature shape:", embeddings.shape)  # (N, 1024)


# %%

import faiss

index = faiss.IndexFlatIP(1024)  # cosine similarity
index.add(embeddings)

# 查詢第 0 張影像最相似的 5 張
query = embeddings[0:1]
D, I = index.search(query, 5)

print("Query:", image_paths[0])
print("Top 5 similar images:")
for i, score in zip(I[0], D[0]):
    print(f"  {image_paths[i]}  (similarity={score:.4f})")


# %%
import matplotlib.pyplot as plt

def show_similar_images(query_idx, top_k=5):
    fig, axes = plt.subplots(1, top_k, figsize=(15, 4))
    for j, ax in enumerate(axes):
        img = Image.open(image_paths[I[0][j]])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{D[0][j]:.2f}")
    plt.show()

# for i in range(20):
#     show_similar_images(i, top_k=5)

similarity_matrix = np.dot(embeddings, embeddings.T)


# %%
import networkx as nx

# === 參數設定 ===
threshold = 0.8  # 相似度閾值，越高連線越少越準確
top_k = 5        # 每個節點連線的最多數量

# === 建立圖 ===
G = nx.Graph()

# 加入節點（每張圖片）
for idx, path in enumerate(image_paths):
    G.add_node(idx, label=os.path.basename(path))

# 根據相似度矩陣加入邊
N = len(image_paths)
for i in range(N):
    # 取 top-k 最相似（排除自己）
    sims = similarity_matrix[i]
    top_idx = sims.argsort()[::-1][1:top_k+1]
    for j in top_idx:
        if sims[j] > threshold:
            G.add_edge(i, j, weight=sims[j])

print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# %%
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# 節點大小與顏色
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.8)
nx.draw_networkx_labels(G, pos, labels={i: i for i in G.nodes()}, font_size=8)

# 邊根據相似度著色
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges,
                       width=[2*w for w in weights],
                       edge_color=weights, edge_cmap=plt.cm.plasma)

plt.title("Scene Graph based on DINOv2 Similarity")
plt.axis('off')
plt.show()



# %%
# 如果圖不連通，只取最大那個子圖
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"⚠️ Graph not connected, using largest connected component with {len(G.nodes())} nodes.")


# %%
import networkx as nx

# 把相似度轉成距離（越相似距離越短）
for u, v, d in G.edges(data=True):
    d['distance'] = 1 - d['weight']

# 使用 NetworkX 近似旅行推銷員 (TSP) 路徑
from networkx.algorithms import approximation as approx

tsp_path = approx.traveling_salesman_problem(
    G, 
    weight='distance', 
    cycle=False  # cycle=False 表示不需要回到起點
)

print("📍 Hamiltonian-like Path covering all nodes:")
print(" -> ".join([str(p) for p in tsp_path]))


# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# 根據路徑順序取得照片
ordered_images = [Image.open(image_paths[i]).convert("RGB") for i in tsp_path]

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(ordered_images[0])
ax.axis("off")
plt.title("Scene Path Traversal (DINOv2 + TSP)")

def update(frame):
    im.set_array(ordered_images[frame])
    ax.set_title(f"Step {frame+1}/{len(ordered_images)} | Image #{tsp_path[frame]}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(ordered_images), interval=800, blit=True)
plt.show()



