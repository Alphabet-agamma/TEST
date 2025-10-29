import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from itertools import combinations
from tqdm import tqdm

# -----------------------------
# ⚙️ 설정
# -----------------------------
file_path = "tourism_clusters_hierarchical_final_800m.csv"
output_excel = "tourism_clusters_hierarchical_final_800m_with_centrality.xlsx"

# -----------------------------
# 1️⃣ 데이터 불러오기
# -----------------------------
data = pd.read_csv(file_path)
print(f"✅ 데이터 로드 완료: {data.shape}")

# -----------------------------
# 2️⃣ 군집별 중심점(centroid) 계산
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
# 3️⃣ 도로망 불러오기 (서울 전체 대신 분석영역 한정)
# -----------------------------
north = data["latitude"].max() + 0.01
south = data["latitude"].min() - 0.01
east = data["longitude"].max() + 0.01
west = data["longitude"].min() - 0.01

print("🛣️ 도로망 불러오는 중 (bbox 기준)...")
G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive")
print(f"✅ 도로망 노드 수: {len(G.nodes)}, 엣지 수: {len(G.edges)}")

# -----------------------------
# 4️⃣ 군집 중심 간 최단 거리 계산
# -----------------------------
edges = []
for (i, row1), (j, row2) in tqdm(
    combinations(clusters.iterrows(), 2),
    total=len(clusters) * (len(clusters) - 1) // 2,
    desc="군집 간 최단 거리 계산 중"
):
    try:
        orig = ox.distance.nearest_nodes(G, row1["centroid_lon"], row1["centroid_lat"])
        dest = ox.distance.nearest_nodes(G, row2["centroid_lon"], row2["centroid_lat"])
        dist = nx.shortest_path_length(G, orig, dest, weight="length")

        # 도로망 상 거리 10km 이하인 경우만 네트워크 간선으로 추가
        if dist <= 10000:
            edges.append((row1["cluster_id"], row2["cluster_id"], dist))
    except nx.NetworkXNoPath:
        continue

print(f"✅ 연결된 군집 쌍 수: {len(edges)}")

# -----------------------------
# 5️⃣ 군집 네트워크 구성
# -----------------------------
G_cluster = nx.Graph()
for a, b, dist in edges:
    G_cluster.add_edge(a, b, weight=dist)

print(f"✅ 네트워크 노드 수: {len(G_cluster.nodes())}, 엣지 수: {len(G_cluster.edges())}")

# -----------------------------
# 6️⃣ 중심성 계산
# -----------------------------
print("📊 중심성 계산 중...")

# Closeness Centrality (다른 군집으로의 접근 용이성)
closeness = nx.closeness_centrality(G_cluster, distance="weight")

# Betweenness Centrality (다른 관광권역 간 이동 매개 정도)
betweenness = nx.betweenness_centrality(G_cluster, weight="weight", normalized=True)

# Eigenvector Centrality (군집 매력도 반영)
# 각 노드의 매력도 = total_places × avg_review_rate
attr_weight = {
    row["cluster_id"]: row["total_places"] * row["avg_review_rate"]
    for _, row in clusters.iterrows()
}

# 간선 가중치: 거리의 역수 × 연결된 군집 매력도 평균
for u, v, d in G_cluster.edges(data=True):
    w_u = attr_weight.get(u, 1)
    w_v = attr_weight.get(v, 1)
    G_cluster[u][v]["weight"] = (w_u + w_v) / 2 / (d["weight"] + 1)

# Eigenvector 중심성 계산
eigenvector = nx.eigenvector_centrality_numpy(G_cluster, weight="weight")

# -----------------------------
# 7️⃣ 결과 병합
# -----------------------------
clusters["centroid_lat"] = clusters["centroid_lat"].round(6)
clusters["centroid_lon"] = clusters["centroid_lon"].round(6)
clusters["Closeness"] = clusters["cluster_id"].map(closeness)
clusters["Betweenness"] = clusters["cluster_id"].map(betweenness)
clusters["Eigenvector"] = clusters["cluster_id"].map(eigenvector)

# -----------------------------
# 8️⃣ 기존 데이터와 병합
# -----------------------------
merged = data.merge(
    clusters[["cluster_id", "centroid_lat", "centroid_lon", "Closeness", "Betweenness", "Eigenvector"]],
    on="cluster_id",
    how="left"
)

# -----------------------------
# 9️⃣ 결과 저장
# -----------------------------
merged.to_excel(output_excel, index=False)
print(f"✅ 결과 저장 완료: {output_excel}")

# -----------------------------
# 🔍 주요 결과 미리보기
# -----------------------------
print("\n📄 결과 미리보기:")
print(merged.head(10))
