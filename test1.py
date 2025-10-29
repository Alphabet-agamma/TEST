import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from math import radians
from tqdm import tqdm

# -----------------------------
# 설정
# -----------------------------
files = [
    "clean_restaurant_geocoded.csv",
    "clean_lodgement.csv"
]

eps_meters = 1000  # 군집 반경 (단위: m)
min_samples = 3    # 최소 군집 크기

output_file = "tourism_clusters.csv"

# -----------------------------
# 1️⃣ 여러 CSV 통합
# -----------------------------
dfs = []
for f in files:
    df = pd.read_csv(f)
    df['category'] = f.split("_")[1]  # 파일명 기준 카테고리 추정 (예: restaurant, lodgement)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"✅ 통합된 데이터 크기: {data.shape}")

# 필요한 컬럼만 남기기 (이름, 좌표, 평점, 카테고리)
cols = [col for col in data.columns if col.lower() in ['name', 'latitude', 'longitude', 'review rate', 'review_rate', 'category']]
data = data[cols]
data.columns = ['name', 'latitude', 'longitude', 'review_rate', 'category']

# 결측치 제거
data = data.dropna(subset=['latitude', 'longitude'])
print(f"✅ 좌표 결측 제거 후: {data.shape[0]}개")

# -----------------------------
# 2️⃣ 좌표 → 라디안 변환 (haversine 거리 계산용)
# -----------------------------
coords = data[['latitude', 'longitude']].to_numpy()
coords_rad = np.radians(coords)

# DBSCAN 파라미터: eps를 라디안 거리로 변환
eps_rad = eps_meters / 6371000.0  # 지구 반지름(m)

# -----------------------------
# 3️⃣ 군집 생성 (DBSCAN)
# -----------------------------
print("📍 DBSCAN 군집화 중...")
db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit(coords_rad)
data['cluster_id'] = db.labels_

# -----------------------------
# 4️⃣ 군집 통계 요약
# -----------------------------
cluster_summary = (
    data.groupby('cluster_id')
    .agg(
        total_places=('name', 'count'),
        avg_review_rate=('review_rate', 'mean'),
        num_categories=('category', lambda x: x.nunique()),
        top_categories=('category', lambda x: x.value_counts().head(3).index.tolist())
    )
    .reset_index()
)

# -----------------------------
# 5️⃣ 결과 병합 및 저장
# -----------------------------
result = data.merge(cluster_summary, on='cluster_id', how='left')
result.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 군집 생성 완료 및 저장: {output_file}")
print(result.head())
