import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from math import radians, sin, cos, asin, sqrt

# -----------------------------
# 설정
# -----------------------------
files = [
    "clean_restaurant_geocoded.csv",
    "clean_lodgement.csv"
]

max_distance = 2000  # 군집 내 최대 거리 (m)
output_file = "tourism_clusters_hierarchical_2km.csv"

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
data["cluster_id"] = labels

print(f"✅ 총 군집 수: {len(set(labels))}")

# -----------------------------
# 4️⃣ 군집 요약 통계
# -----------------------------
cluster_summary = (
    data.groupby("cluster_id")
    .agg(
        total_places=("name", "count"),
        avg_review_rate=("review_rate", "mean"),
        num_categories=("category", "nunique"),
        top_categories=("category", lambda x: x.value_counts().head(3).index.tolist())
    )
    .reset_index()
)

# -----------------------------
# 5️⃣ 결과 저장
# -----------------------------
# cluster_id 타입 통일
data["cluster_id"] = data["cluster_id"].astype(int)
cluster_summary["cluster_id"] = cluster_summary["cluster_id"].astype(int)

# 병합
result = data.merge(cluster_summary, on="cluster_id", how="left")
result = data.merge(cluster_summary, on="cluster_id", how="left")
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ 군집 생성 완료 및 저장: {output_file}")
print(f"총 군집 수: {len(cluster_summary)}")
print(result.head(10))
