from flask import Flask, request, jsonify
import numpy as np
from data_loader import load_data
from recommender import recommend_movies
from flask_cors import CORS
from rl_model import update
from sentiment import analyze_review

app = Flask(__name__)
CORS(app)

df, user_movie_matrix, movie_map, movie_poster_map = load_data()

@app.route("/popular", methods=["GET"])
def get_popular():
    movie_stats = df.groupby("movieId").agg({"rating": ["mean", "count"]})
    movie_stats.columns = ["avg_rating", "num_ratings"]
    movie_stats["popularity"] = (
        movie_stats["avg_rating"] * np.log1p(movie_stats["num_ratings"])
    )
    top = movie_stats.sort_values(by="popularity", ascending=False).head(20)

    result = []
    for movie_id in top.index:
        title = movie_map[movie_id]
        result.append({
            "movieId": int(movie_id),
            "title": title,
            "rating": float(movie_stats.loc[movie_id]["avg_rating"]),
            "poster": movie_poster_map.get(movie_id),
        })

    return jsonify(result)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    selected_ids = data.get("movies", [])

    recs = recommend_movies(user_movie_matrix, selected_ids, movie_map, df=df)

    for rec in recs:
        rec["poster"] = movie_poster_map.get(rec["movieId"])

    return jsonify({"recommendations": recs})

@app.route("/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify([])

    results = df[df["title"].str.lower().str.contains(query)]
    results = results.drop_duplicates(subset="movieId").head(10)

    output = []
    for _, row in results.iterrows():
        output.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
        })

    return jsonify(output)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    movie_id = data.get("movieId")
    clicked = data.get("clicked", True)
    update(movie_id, clicked)
    return jsonify({"status": "updated"})

@app.route("/review", methods=["POST"])
def review():
    data = request.json
    movie_id = data.get("movieId")
    review_text = data.get("review")

    score = analyze_review(movie_id, review_text)

    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return jsonify({
        "status": "review added",
        "sentiment": {
            "score": round(score, 3),
            "label": label,
        }
    })

def handler(request):
    return app(request)
