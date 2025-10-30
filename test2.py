import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from itertools import combinations
from tqdm import tqdm
import folium

# -----------------------------
# ⚙️ 설정
# -----------------------------
file_path = "tourism_clusters_hierarchical_final_800m.csv"
output_excel = "tourism_clusters_hierarchical_final_800m_with_full_centrality_filtered.xlsx"
output_map = "tourism_network_map_clusters_filtered_800m.html"

# 서울 경계 (대략)
SEOUL_BOUND = {
    "north": 37.715,
    "south": 37.413,
    "west": 126.734,
    "east": 127.183
}

# -----------------------------
# 1️⃣ 데이터 불러오기
# -----------------------------
data = pd.read_csv(file_path)
print(f"✅ 데이터 로드 완료: {data.shape}")

# -----------------------------
# 2️⃣ 군집 중심 계산
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

print(f"📍 원래 군집 수: {len(clusters)}")

# -----------------------------
# 3️⃣ 서울 내부 군집만 유지
# -----------------------------
clusters = clusters[
    (clusters["centroid_lat"] >= SEOUL_BOUND["south"]) &
    (clusters["centroid_lat"] <= SEOUL_BOUND["north"]) &
    (clusters["centroid_lon"] >= SEOUL_BOUND["west"]) &
    (clusters["centroid_lon"] <= SEOUL_BOUND["east"])
].reset_index(drop=True)

print(f"✅ 서울 내부 군집만 유지: {len(clusters)}개")

# -----------------------------
# 4️⃣ 도로망 불러오기 (drive_service)
# -----------------------------
north = SEOUL_BOUND["north"] + 0.01
south = SEOUL_BOUND["south"] - 0.01
east = SEOUL_BOUND["east"] + 0.01
west = SEOUL_BOUND["west"] - 0.01

print("🛣️ 도로망 불러오는 중...")
G = ox.graph_from_bbox((north, south, east, west), network_type="drive_service", simplify=True)
print(f"✅ 도로망 노드 수: {len(G.nodes)}, 엣지 수: {len(G.edges)}")

# -----------------------------
# 5️⃣ 군집 중심 간 최단거리 계산 (전체 쌍)
# -----------------------------
edges = []
print("📏 군집 간 최단거리 계산 중...")

for (i, row1), (j, row2) in tqdm(
    combinations(clusters.iterrows(), 2),
    total=len(clusters) * (len(clusters) - 1) // 2
):
    try:
        orig = ox.distance.nearest_nodes(G, row1["centroid_lon"], row1["centroid_lat"])
        dest = ox.distance.nearest_nodes(G, row2["centroid_lon"], row2["centroid_lat"])
        dist = nx.shortest_path_length(G, orig, dest, weight="length")
        edges.append((row1["cluster_id"], row2["cluster_id"], dist))
    except nx.NetworkXNoPath:
        edges.append((row1["cluster_id"], row2["cluster_id"], np.inf))

print(f"✅ 계산 완료: {len(edges)} 쌍")

# -----------------------------
# 6️⃣ 네트워크 생성 (전체 연결 그래프)
# -----------------------------
G_cluster = nx.Graph()

# 노드 추가
for _, row in clusters.iterrows():
    G_cluster.add_node(
        row["cluster_id"],
        attr_weight=row["total_places"] * row["avg_review_rate"],
        lat=row["centroid_lat"],
        lon=row["centroid_lon"]
    )

# 엣지 추가 (무한 거리 제외)
for a, b, dist in edges:
    if np.isfinite(dist):
        G_cluster.add_edge(a, b, weight=dist)

print(f"✅ 그래프 구성 완료: 노드 {len(G_cluster.nodes())}, 엣지 {len(G_cluster.edges())}")

# -----------------------------
# 7️⃣ 중심성 계산
# -----------------------------
print("📊 중심성 계산 중...")

# Closeness (거리 기반 접근성)
closeness = nx.closeness_centrality(G_cluster, distance="weight")

# Betweenness (경로 매개 중심성)
betweenness = nx.betweenness_centrality(G_cluster, weight="weight", normalized=True)

# Eigenvector (매력도 가중 중심성)
for u, v, d in G_cluster.edges(data=True):
    w_u = G_cluster.nodes[u]["attr_weight"]
    w_v = G_cluster.nodes[v]["attr_weight"]
    G_cluster[u][v]["weight"] = (w_u + w_v) / 2 / (d["weight"] + 1)

eigenvector = nx.eigenvector_centrality_numpy(G_cluster, weight="weight")

# -----------------------------
# 8️⃣ 결과 병합
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
# 9️⃣ 결과 저장
# -----------------------------
merged.to_excel(output_excel, index=False)
print(f"✅ 중심성 계산 완료 및 저장: {output_excel}")

# -----------------------------
# 🔟 지도 시각화 (Folium)
# -----------------------------
print("🗺️ 지도 시각화 중...")

map_center = [clusters["centroid_lat"].mean(), clusters["centroid_lon"].mean()]
m = folium.Map(location=map_center, zoom_start=11, tiles="cartodb positron")

max_ev = clusters["Eigenvector"].max()

# 마커 추가
for _, row in clusters.iterrows():
    radius = 4 + 30 * (row["Eigenvector"] / max_ev if max_ev > 0 else 0)
    popup_html = f"""
    <b>Cluster {row['cluster_id']}</b><br>
    장소 수: {row['total_places']}<br>
    평균 평점: {row['avg_review_rate']:.2f}<br>
    Closeness: {row['Closeness']:.4f}<br>
    Betweenness: {row['Betweenness']:.6f}<br>
    Eigenvector: {row['Eigenvector']:.6f}
    """
    folium.CircleMarker(
        location=[row["centroid_lat"], row["centroid_lon"]],
        radius=radius,
        color="crimson",
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# 중심 간 연결선
for a, b, dist in edges:
    if np.isfinite(dist):
        lat1 = clusters.loc[clusters["cluster_id"] == a, "centroid_lat"].values[0]
        lon1 = clusters.loc[clusters["cluster_id"] == a, "centroid_lon"].values[0]
        lat2 = clusters.loc[clusters["cluster_id"] == b, "centroid_lat"].values[0]
        lon2 = clusters.loc[clusters["cluster_id"] == b, "centroid_lon"].values[0]
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="gray", weight=1, opacity=0.3).add_to(m)

m.save(output_map)
print(f"✅ 관광지 군집 네트워크 지도 저장 완료: {output_map}")

print("\n📄 미리보기:")
print(merged.head(10))
