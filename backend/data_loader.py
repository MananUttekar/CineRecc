import pandas as pd
import os

#absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_path = os.path.join(BASE_DIR, "data", "movies.csv")
ratings_path = os.path.join(BASE_DIR, "data", "ratings.csv")

def load_data():
    print("Movies path:", movies_path)
    print("Ratings path:", ratings_path)

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    df = pd.merge(ratings, movies, on="movieId")
    movie_poster_map = dict(zip(movies["movieId"], movies.get("poster_path", [""] * len(movies))))
    user_movie_matrix = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    movie_map = dict(zip(movies["movieId"], movies["title"]))

    return df, user_movie_matrix, movie_map, movie_poster_map