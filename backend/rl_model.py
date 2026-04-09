import random

#Store data in memory
bandit_data = {}


def get_score(movie_id):
    if movie_id not in bandit_data:
        bandit_data[movie_id] = {"views": 0, "clicks": 0}

    data = bandit_data[movie_id]

    if data["views"] == 0:
        return 0

    return data["clicks"] / data["views"]


def select_movies(candidates, top_n=10, epsilon=0.1):
    candidates = candidates.copy()
    selected = []

    for _ in range(min(top_n, len(candidates))):
        if random.random() < epsilon:
            movie = random.choice(candidates)   # explore
        else:
            movie = max(candidates, key=get_score)  # exploit

        selected.append(movie)
        candidates.remove(movie)

        #Record an impression when a movie is surfaced to the user
        if movie not in bandit_data:
            bandit_data[movie] = {"views": 0, "clicks": 0}
        bandit_data[movie]["views"] += 1

    return selected


def update(movie_id, clicked=True):
    #Call this only to record an explicit click/thumbs-up
    if movie_id not in bandit_data:
        bandit_data[movie_id] = {"views": 0, "clicks": 0}

    if clicked:
        bandit_data[movie_id]["clicks"] += 1
