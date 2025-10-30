import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from itertools import combinations
from tqdm import tqdm
import os

# -----------------------------
# ⚙️ 설정
# -----------------------------
file_path = "tourism_clusters_hierarchical_final_800m.csv"
output_excel = "tourism_clusters_hierarchical_final_800m_with_centrality.xlsx"
graphml_path = "seoul_drive.graphml"  # 도로망 캐시 파일

# -----------------------------
# 1️⃣ 데이터 불러오기
# -----------------------------
data = pd.read_csv(file_path)
print(f"✅ 데이터 로드 완료: {data.shape}")

# -----------------------------
# 2️⃣ 군집 중심 계산 (centroid)
# -----------------------------
clusters = (
    data.groupby("cluster_id")
    .agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        total_places=("total_places", "first"),
        avg_review_rate=("avg_review_rate", "first")
    )
    .reset_index()
)

print(f"✅ 군집 수: {len(clusters)}")

# -----------------------------
# 3️⃣ 서울 도로망 불러오기 (캐시 방식)
# -----------------------------
if os.path.exists(graphml_path):
    print("📂 캐시된 서울 도로망 불러오는 중...")
    G = ox.load_graphml(graphml_path)
else:
    print("🌐 OSM에서 서울 도로망 불러오는 중... (약 1~2분 소요)")
    G = ox.graph_from_place("Seoul, South Korea", network_type="drive")
    ox.save_graphml(G, graphml_path)
    print(f"💾 도로망 저장 완료: {graphml_path}")

print(f"✅ 도로망 노드 수: {len(G.nodes)}, 엣지 수: {len(G.edges)}")

# -----------------------------
# 4️⃣ 군집 중심 간 최단거리 계산
# -----------------------------
edges = []
print("📏 군집 간 최단거리 계산 중... (모든 쌍 비교)")

for (i, row1), (j, row2) in tqdm(
    combinations(clusters.iterrows(), 2),
    total=len(clusters) * (len(clusters) - 1) // 2,
    desc="군집 간 거리 계산"
):
    try:
        orig = ox.distance.nearest_nodes(G, row1["centroid_lon"], row1["centroid_lat"])
        dest = ox.distance.nearest_nodes(G, row2["centroid_lon"], row2["centroid_lat"])
        dist = nx.shortest_path_length(G, orig, dest, weight="length")
        edges.append((row1["cluster_id"], row2["cluster_id"], dist))
    except nx.NetworkXNoPath:
        # 도로망으로 연결 안 되는 경우 무한 거리로 표시
        edges.append((row1["cluster_id"], row2["cluster_id"], np.inf))

print(f"✅ 거리 계산 완료: {len(edges)} 쌍")

# -----------------------------
# 5️⃣ 네트워크 구성
# -----------------------------
G_cluster = nx.Graph()

# 노드 추가 (매력도 가중치 포함)
for _, row in clusters.iterrows():
    G_cluster.add_node(
        row["cluster_id"],
        attr_weight=row["total_places"] * row["avg_review_rate"]
    )

# 엣지 추가 (무한 거리 제외)
for a, b, dist in edges:
    if np.isfinite(dist):
        G_cluster.add_edge(a, b, weight=dist)

print(f"✅ 네트워크 구성 완료: 노드 {len(G_cluster.nodes())}, 엣지 {len(G_cluster.edges())}")

# -----------------------------
# 6️⃣ 중심성 계산
# -----------------------------
print("📊 중심성 계산 중...")

# Closeness Centrality (다른 군집까지의 평균 거리 기반 접근성)
closeness = nx.closeness_centrality(G_cluster, distance="weight")

# Betweenness Centrality (경로상 매개 역할 정도)
betweenness = nx.betweenness_centrality(G_cluster, weight="weight", normalized=True)

# Eigenvector Centrality (군집 매력도 반영)
# 간선 가중치 = 거리의 역수 × 연결된 군집 매력도 평균
for u, v, d in G_cluster.edges(data=True):
    w_u = G_cluster.nodes[u]["attr_weight"]
    w_v = G_cluster.nodes[v]["attr_weight"]
    G_cluster[u][v]["weight"] = (w_u + w_v) / 2 / (d["weight"] + 1)

eigenvector = nx.eigenvector_centrality_numpy(G_cluster, weight="weight")

print("✅ 중심성 계산 완료")

# -----------------------------
# 7️⃣ 결과 병합
# -----------------------------
clusters["Closeness"] = clusters["cluster_id"].map(closeness)
clusters["Betweenness"] = clusters["cluster_id"].map(betweenness)
clusters["Eigenvector"] = clusters["cluster_id"].map(eigenvector)

merged = data.merge(
    clusters[
        ["cluster_id", "centroid_lat", "centroid_lon", "Closeness", "Betweenness", "Eigenvector"]
    ],
    on="cluster_id",
    how="left"
)

# -----------------------------
# 8️⃣ 결과 저장
# -----------------------------
merged.to_excel(output_excel, index=False)
print(f"✅ 결과 저장 완료: {output_excel}")

# -----------------------------
# 9️⃣ 미리보기
# -----------------------------
print("\n📄 결과 미리보기:")
print(merged.head(10))
