import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 불러오기
dataset = pd.read_csv('./data/final_perfume_data.csv', encoding='unicode_escape')

# 피처 벡터 생성
features = dataset['Description'] + ' ' + dataset['Notes']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(features.values.astype('U'))

# 유사도 매트릭스 계산
similarity_matrix = cosine_similarity(feature_vectors, feature_vectors)

# 향수 추천 함수
def recommend_perfumes(liked_perfumes, top_n=5):
    # 사용자가 좋아하는 향수의 인덱스 찾기
    liked_indices = dataset[dataset['Name'].isin(liked_perfumes)].index

    # 좋아하는 향수의 유사도 점수 계산
    similarity_scores = similarity_matrix[liked_indices]

    # 평균 유사도 점수 계산 (좋아하는 향수 제외)
    average_similarity = similarity_scores.mean(axis=0)
    average_similarity[liked_indices] = 0  # 좋아하는 향수의 유사도 점수를 0으로 설정

    # 상위 n개의 유사한 향수 인덱스 가져오기
    top_indices = average_similarity.argsort()[::-1][:top_n]

    # 추천 향
