import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os

# -----------------------------
# 설정
# -----------------------------
files = [
    "clean_restaurant_geocoded.csv",
    "clean_lodgement.csv"
]

eps_meters = 2000  # 군집 반경 (단위: m)
min_samples = 10    # 최소 군집 크기
output_file = "tourism_clusters_final.csv"

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
    if 'name' in df.columns:
        rename_dict['name'] = 'name'
    if 'title' in df.columns:
        rename_dict['title'] = 'name'

    df = df.rename(columns=rename_dict)

    # 🔧 파일명 기반 카테고리 자동 추출
    base_name = os.path.basename(f).lower()  # 파일명만 추출, 소문자화
    if "restaurant" in base_name:
        category_name = "restaurant"
    elif "lodgement" in base_name or "lodging" in base_name or "hotel" in base_name:
        category_name = "lodgement"
    else:
        # 혹시 다른 파일 추가될 경우 대비
        category_name = os.path.splitext(base_name)[0].replace("clean_", "")
    df['category'] = category_name

    dfs.append(df)

# 🔹 통합
data = pd.concat(dfs, ignore_index=True)
print(f"✅ 통합된 데이터 크기: {data.shape}")

# -----------------------------
# 2️⃣ 필요한 컬럼만 선택
# -----------------------------
# 없는 컬럼은 자동 추가
for col in ['name', 'latitude', 'longitude', 'review_rate', 'category']:
    if col not in data.columns:
        data[col] = np.nan

data = data[['name', 'latitude', 'longitude', 'review_rate', 'category']]

# 결측 제거
data = data.dropna(subset=['latitude', 'longitude'])
print(f"✅ 좌표 결측 제거 후: {data.shape[0]}개")

# -----------------------------
# 3️⃣ 좌표 → 라디안 변환 (Haversine 거리 계산용)
# -----------------------------
coords = data[['latitude', 'longitude']].to_numpy()
coords_rad = np.radians(coords)

# DBSCAN 거리 반경 (라디안)
eps_rad = eps_meters / 6371000.0  # 지구 반지름(m)

# -----------------------------
# 4️⃣ DBSCAN 군집화
# -----------------------------
print("📍 DBSCAN 군집화 중...")
db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit(coords_rad)
data['cluster_id'] = db.labels_

# -----------------------------
# 5️⃣ 군집 통계 요약
# -----------------------------
cluster_summary = (
    data.groupby('cluster_id')
    .agg(
        total_places=('name', 'count'),
        avg_review_rate=('review_rate', 'mean'),
        num_categories=('category', 'nunique'),
        top_categories=('category', lambda x: x.value_counts().head(3).index.tolist())
    )
    .reset_index()
)

# -----------------------------
# 6️⃣ 결과 병합 및 저장
# -----------------------------
result = data.merge(cluster_summary, on='cluster_id', how='left')
result.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 군집 생성 완료 및 저장: {output_file}")
print(result.head(10))
