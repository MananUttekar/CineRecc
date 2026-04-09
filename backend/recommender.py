import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rl_model import select_movies
from sentiment import get_sentiment

def _popularity_fallback(df, movie_map, exclude_ids, top_n):
    stats = df.groupby("movieId").agg({"rating": ["mean", "count"]})
    stats.columns = ["avg_rating", "num_ratings"]
    stats["score"] = stats["avg_rating"] * np.log1p(stats["num_ratings"])
    stats = stats[~stats.index.isin(exclude_ids)]
    top = stats.nlargest(top_n, "score")
    return [
        {"movieId": int(mid), "title": movie_map[mid], "fallback": True}
        for mid in top.index if mid in movie_map
    ]

def recommend_movies(user_movie_matrix, selected_ids, movie_map, df=None, top_n=10):

    matrix = user_movie_matrix.copy()

    known_ids = [mid for mid in selected_ids if mid in matrix.columns]

    if not known_ids:
        print("[recommender] Full cold start — falling back to popularity.")
        if df is not None:
            return _popularity_fallback(df, movie_map, selected_ids, top_n)
        return []

    pseudo_user = pd.Series(index=matrix.columns, dtype=float)
    for movie_id in known_ids:
        pseudo_user[movie_id] = 5.0

    matrix.loc["new_user"] = pseudo_user

    filled = matrix.fillna(0)
    similarity = cosine_similarity(filled)

    user_index = list(matrix.index).index("new_user")
    sim_scores = similarity[user_index]

    similar_users = sim_scores.argsort()[::-1][1:11]

    cf_scores = {}
    for user in similar_users:
        sim = sim_scores[user]
        user_ratings = matrix.iloc[user]
        for movie_id, rating in user_ratings.items():
            if pd.isna(rating):
                continue
            if movie_id in selected_ids:
                continue
            cf_scores[movie_id] = cf_scores.get(movie_id, 0) + sim * rating

    if not cf_scores:
        print("[recommender] CF produced no candidates — falling back to popularity.")
        if df is not None:
            return _popularity_fallback(df, movie_map, selected_ids, top_n)
        return []

    max_cf   = max(cf_scores.values())
    min_cf   = min(cf_scores.values())
    cf_range = max_cf - min_cf if max_cf != min_cf else 1.0

    SENTIMENT_WEIGHT = 0.15

    blended = {}
    for movie_id, raw_cf in cf_scores.items():
        norm_cf   = (raw_cf - min_cf) / cf_range
        sentiment = get_sentiment(movie_id)
        blended[movie_id] = norm_cf + SENTIMENT_WEIGHT * sentiment

    recommended = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    candidates  = [mid for mid, _ in recommended[:50]]
    final_ids   = select_movies(candidates, top_n=top_n)

    return [
        {"movieId": int(mid), "title": movie_map[mid], "fallback": False}
        for mid in final_ids
        if mid in movie_map
    ]
