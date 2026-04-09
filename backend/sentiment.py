from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

#Store sentiment scores per movie
sentiment_data = {}

def analyze_review(movie_id, review):
    score = analyzer.polarity_scores(review)["compound"]

    if movie_id not in sentiment_data:
        sentiment_data[movie_id] = []

    sentiment_data[movie_id].append(score)

    return score  

def get_sentiment(movie_id):
    if movie_id not in sentiment_data:
        return 0

    scores = sentiment_data[movie_id]
    return sum(scores) / len(scores)
