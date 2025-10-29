import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from math import radians, sin, cos, asin, sqrt

# -----------------------------
# 설정
# -----------------------------
files = [
    "clean_restaurant_geocoded.csv",
    "clean_lodgement.csv"
]

max_distance = 2000  # 군집 내 최대 거리 (단위: m)
initial_clusters = 50  # 초기 군집 수 (KMeans)
output_file = "tourism_clusters_kmeans_2km.csv"

# -----------------------------
# 1️⃣ 여러 CSV 통합
# -----------------------------
dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]  # 열 이름 소문자 정리

    # 🔧 컬럼명 표준화
    rename_dict = {}
    if 'aggregate rating' in df.columns:
        rename_dict['aggregate rating'] = 'review_rate'
    if 'review rate' in df.columns:
        rename_dict['review rate'] = 'review_rate'
    if 'latitude' in df.columns:
        rename_dict['latitude'] = 'latitude'
    if 'longitude' in df.columns:
        rename_dict['longitude'] = 'longitude'
    if 'lat' in df.columns:
        rename_dict['lat'] = 'latitude'
    if 'lng' in df.columns:
        rename_dict['lng'] = 'longitude'
    if 'name' in df.columns:
        rename_dict['name'] = 'name'
    if 'title' in df.columns:
        rename_dict['title'] = 'name'

    df = df.rename(columns=rename_dict)

    # 🔧 파일명 기반 카테고리 추출
    base_name = os.path.basename(f)
    if "restaurant" in base_name.lower():
        df["category"] = "restaurant"
    elif "lodgement" in base_name.lower():
        df["category"] = "lodgement"
    else:
        df["category"] = "unknown"

    dfs.append(df)

# 🔹 통합
data = pd.concat(dfs, ignore_index=True)
print(f"✅ 통합된 데이터 크기: {data.shape}")

# -----------------------------
# 2️⃣ 필요한 컬럼만 선택
# -----------------------------
for col in ["name", "latitude", "longitude", "review_rate", "category"]:
    if col not in data.columns:
        data[col] = np.nan

data = data[["name", "latitude", "longitude", "review_rate", "category"]]
data = data.dropna(subset=["latitude", "longitude"])
print(f"✅ 좌표 결측 제거 후: {data.shape[0]}개")

# -----------------------------
# 3️⃣ 거리 계산 함수
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구 반지름 (m)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def max_pairwise_distance(coords):
    """군집 내 점들 간 최대 거리 계산"""
    max_dist = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d > max_dist:
                max_dist = d
    return max_dist

# -----------------------------
# 4️⃣ 1차 군집화 (KMeans)
# -----------------------------
coords = data[["latitude", "longitude"]].to_numpy()
print("📍 1차 KMeans 군집화 중...")
kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
data["cluster_id"] = kmeans.fit_predict(coords)

# -----------------------------
# 5️⃣ 2차 거리 기반 보정 (2km 초과 시 분리)
# -----------------------------
print("🔧 군집 내 거리 조건 보정 중...")

adjusted_labels = []
cluster_counter = 0

for cid in tqdm(sorted(data["cluster_id"].unique())):
    cluster_data = data[data["cluster_id"] == cid]
    cluster_coords = cluster_data[["latitude", "longitude"]].to_numpy()

    if len(cluster_coords) <= 1:
        adjusted_labels.extend([cluster_counter] * len(cluster_coords))
        cluster_counter += 1
        continue

    # 군집 내 최대 거리 계산
    max_d = max_pairwise_distance(cluster_coords)

    if max_d <= max_distance:
        # 조건 만족 시 그대로 사용
        adjusted_labels.extend([cluster_counter] * len(cluster_coords))
        cluster_counter += 1
    else:
        # 조건 불만족 → 다시 세부 KMeans 적용
        n_sub = int(np.ceil(max_d / max_distance))
        sub_kmeans = KMeans(n_clusters=min(n_sub, len(cluster_coords)), random_state=42, n_init=10)
        sub_labels = sub_kmeans.fit_predict(cluster_coords)
        for sub in sub_labels:
            adjusted_labels.append(cluster_counter + sub)
        cluster_counter += n_sub

data["cluster_id_final"] = adjusted_labels

# -----------------------------
# 6️⃣ 군집 요약 통계
# -----------------------------
cluster_summary = (
    data.groupby("cluster_id_final")
    .agg(
        total_places=("name", "count"),
        avg_review_rate=("review_rate", "mean"),
        num_categories=("category", "nunique"),
        top_categories=("category", lambda x: x.value_counts().head(3).index.tolist())
    )
    .reset_index()
)

# -----------------------------
# 7️⃣ 결과 저장
# -----------------------------
result = data.merge(cluster_summary, on="cluster_id_final", how="left")
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ 군집 생성 완료 및 저장: {output_file}")
print(f"총 군집 수: {len(cluster_summary)}")
print(result.head(10))
