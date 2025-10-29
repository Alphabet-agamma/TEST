import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from math import radians, sin, cos, asin, sqrt
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

# -----------------------------
# 설정
# -----------------------------
files = [
    "clean_restaurant_geocoded.csv",
    "clean_lodgement.csv"
]

max_distance = 800  # 군집 내 최대 거리 (m)
output_file = "tourism_clusters_hierarchical_final_2km.csv"
output_map = "tourism_clusters_hierarchical_final_2km_map.html"

# -----------------------------
# 1️⃣ 여러 CSV 통합
# -----------------------------
dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]

    # 🔧 컬럼명 표준화
    rename_dict = {}
    if 'aggregate rating' in df.columns:
        rename_dict['aggregate rating'] = 'review_rate'
    if 'review rate' in df.columns:
        rename_dict['review rate'] = 'review_rate'
    if 'lat' in df.columns:
        rename_dict['lat'] = 'latitude'
    if 'lng' in df.columns:
        rename_dict['lng'] = 'longitude'
    if 'title' in df.columns:
        rename_dict['title'] = 'name'
    df = df.rename(columns=rename_dict)

    # 🔧 카테고리 자동 판별
    base_name = os.path.basename(f).lower()
    if "restaurant" in base_name:
        df["category"] = "restaurant"
    elif "lodgement" in base_name:
        df["category"] = "lodgement"
    else:
        df["category"] = "unknown"

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=["latitude", "longitude"])
print(f"✅ 통합된 데이터 크기: {data.shape}")

# -----------------------------
# 2️⃣ 거리 계산 (Haversine)
# -----------------------------
def haversine_pdist(coords):
    """위도/경도 좌표 간 Haversine 거리 행렬 계산 (단위: m)"""
    R = 6371000  # 지구 반지름(m)
    coords_rad = np.radians(coords)

    def pairwise(u, v):
        dlat = v[0] - u[0]
        dlon = v[1] - u[1]
        a = np.sin(dlat / 2)**2 + np.cos(u[0]) * np.cos(v[0]) * np.sin(dlon / 2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    return pdist(coords_rad, metric=pairwise)

coords = data[["latitude", "longitude"]].to_numpy()
print("📏 거리 행렬 계산 중...")
dist_matrix = haversine_pdist(coords)

# -----------------------------
# 3️⃣ 계층적 군집 (complete linkage)
# -----------------------------
print("📍 계층적 군집화 중...")
Z = linkage(dist_matrix, method="complete")

# 2km 이하의 거리 제약으로 군집 자르기
labels = fcluster(Z, t=max_distance, criterion="distance")
data["cluster_id"] = labels.astype(int)

print(f"✅ 총 군집 수: {len(set(labels))}")

# -----------------------------
# 4️⃣ 군집 요약 통계
# -----------------------------
cluster_summary = (
    data.groupby("cluster_id")
    .agg(
        total_places=("name", "count"),
        avg_review_rate=("review_rate", "mean"),
        top_categories=("category", lambda x: x.value_counts().head(3).index.tolist())
    )
    .reset_index()
)

# 카테고리 상위 3개를 각각 분리
for i in range(3):
    cluster_summary[f"category_rank_{i+1}"] = cluster_summary["top_categories"].apply(
        lambda x: x[i] if len(x) > i else np.nan
    )
cluster_summary = cluster_summary.drop(columns=["top_categories"])

# -----------------------------
# 5️⃣ 결과 병합 및 정리
# -----------------------------
result = data.merge(cluster_summary, on="cluster_id", how="left")

# 필요한 컬럼 순서로 정렬
result = result[
    [
        "name",
        "latitude",
        "longitude",
        "category",
        "cluster_id",
        "total_places",
        "avg_review_rate",
        "category_rank_1",
        "category_rank_2",
        "category_rank_3",
    ]
]

# -----------------------------
# 6️⃣ 지도 시각화 (folium)
# -----------------------------
print("🗺️ 지도 시각화 중...")

# 지도 중심은 전체 평균 좌표로 설정
center_lat, center_lon = result["latitude"].mean(), result["longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodb positron")

# 색상 팔레트 설정
unique_clusters = sorted(result["cluster_id"].unique())
colormap = cm.get_cmap("tab20", len(unique_clusters))
norm = colors.Normalize(vmin=0, vmax=len(unique_clusters) - 1)

# 군집별 시각화
for i, cluster_id in enumerate(unique_clusters):
    cluster_data = result[result["cluster_id"] == cluster_id]
    color = colors.to_hex(colormap(norm(i)))

    # 군집 중심점 계산
    center_lat = cluster_data["latitude"].mean()
    center_lon = cluster_data["longitude"].mean()

    # 군집 내 장소 마커 표시
    for _, row in cluster_data.iterrows():
        popup_html = f"""
        <b>{row['name']}</b><br>
        📍 <b>Category:</b> {row['category']}<br>
        🏷️ <b>Cluster:</b> {row['cluster_id']}<br>
        ⭐ <b>Review Rate:</b> {row['avg_review_rate']:.2f}<br>
        👥 <b>Total Places:</b> {row['total_places']}<br>
        🥇 {row['category_rank_1'] or '-'}, 🥈 {row['category_rank_2'] or '-'}, 🥉 {row['category_rank_3'] or '-'}
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    # 군집 내 모든 점을 선으로 연결
    coords = list(zip(cluster_data["latitude"], cluster_data["longitude"]))
    if len(coords) > 1:
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=2,
            opacity=0.5
        ).add_to(m)

print(f"✅ 총 {len(unique_clusters)}개 군집 시각화 완료")

# -----------------------------
# 7️⃣ 저장
# -----------------------------
result.to_csv(output_file, index=False, encoding="utf-8-sig")
m.save(output_map)

print(f"\n✅ 군집 생성 및 지도 저장 완료!")
print(f"📄 데이터: {output_file}")
print(f"🗺️ 지도: {output_map}")
