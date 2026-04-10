import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

function StarRating({ rating }) {
  const stars = Math.round(rating * 2) / 2;
  return (
    <span className="stars">
      {[1, 2, 3, 4, 5].map((s) => (
        <span key={s} className={s <= stars ? "star filled" : s - 0.5 === stars ? "star half" : "star"}>★</span>
      ))}
      <span className="rating-num">{rating.toFixed(1)}</span>
    </span>
  );
}

const SENTIMENT_CONFIG = {
  positive: { emoji: "😊", color: "#4ade80", label: "Positive" },
  neutral:  { emoji: "😐", color: "#facc15", label: "Neutral"  },
  negative: { emoji: "😞", color: "#f87171", label: "Negative" },
};

function SentimentBadge({ sentiment }) {
  const cfg = SENTIMENT_CONFIG[sentiment.label] || SENTIMENT_CONFIG.neutral;
  const pct = Math.round(Math.abs(sentiment.score) * 100);
  return (
    <div className="sentiment-badge" style={{ "--s-color": cfg.color }}>
      <span className="s-emoji">{cfg.emoji}</span>
      <div className="s-info">
        <span className="s-label" style={{ color: cfg.color }}>{cfg.label}</span>
        <div className="s-bar-track">
          <div className="s-bar-fill" style={{ width: `${pct}%`, background: cfg.color }} />
        </div>
      </div>
      <span className="s-score" style={{ color: cfg.color }}>
        {sentiment.score > 0 ? "+" : ""}{sentiment.score.toFixed(2)}
      </span>
    </div>
  );
}

function MovieCard({ movie, onFeedback, variant = "popular" }) {
  const [review, setReview] = useState("");
  const [sentiment, setSentiment] = useState(null);
  const [liked, setLiked] = useState(false);
  const [imgErr, setImgErr] = useState(false);

  const handleFeedback = () => {
    setLiked(true);
    onFeedback(movie.movieId);
  };

  const handleReview = async () => {
    if (!review.trim()) return;
    const res = await fetch(`${API}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movieId: movie.movieId, review }),
    });
    const data = await res.json();
    setSentiment(data.sentiment);
    setReview("");
  };

  const posterUrl = movie.poster && !imgErr
    ? `https://image.tmdb.org/t/p/w342${movie.poster}`
    : null;

  return (
    <div className={`movie-card ${variant}`}>
      <div className="card-poster">
        {posterUrl ? (
          <img
            src={posterUrl}
            alt={movie.title}
            onError={() => setImgErr(true)}
          />
        ) : (
          <div className="poster-placeholder">
            <span>🎬</span>
            <span className="poster-title-fallback">{movie.title}</span>
          </div>
        )}
      </div>
      <div className="card-body">
        <div className="card-title">{movie.title}</div>
        {movie.rating && <StarRating rating={movie.rating} />}
        <div className="card-actions">
          <button
            className={`btn-like ${liked ? "active" : ""}`}
            onClick={handleFeedback}
          >
            {liked ? "❤️" : "🤍"} Like
          </button>
        </div>

        <div className="review-row">
          <input
            className="review-input"
            placeholder="Write a review..."
            value={review}
            onChange={(e) => setReview(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleReview()}
          />
          <button className="btn-submit" onClick={handleReview}>→</button>
        </div>

        {sentiment && <SentimentBadge sentiment={sentiment} />}
      </div>
    </div>
  );
}

export default function App() {
  const [popular, setPopular] = useState([]);
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selected, setSelected] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const searchRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/popular`)
      .then((r) => r.json())
      .then(setPopular)
      .catch(console.error);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleSearch = async (value) => {
    setSearch(value);
    if (value.length < 2) { setSearchResults([]); setShowDropdown(false); return; }
    const res = await fetch(`${API}/search?q=${encodeURIComponent(value)}`);
    const data = await res.json();
    setSearchResults(data);
    setShowDropdown(true);
  };

  const selectMovie = (movie) => {
    if (!selected.find((m) => m.movieId === movie.movieId)) {
      setSelected((prev) => [...prev, movie]);
    }
    setSearch("");
    setSearchResults([]);
    setShowDropdown(false);
  };

  const removeMovie = (movieId) => {
    setSelected((prev) => prev.filter((m) => m.movieId !== movieId));
  };

  const getRecommendations = async () => {
    if (!selected.length) return;
    setLoading(true);
    setRecommendations([]);
    const ids = selected.map((m) => m.movieId);
    const res = await fetch(`${API}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movies: ids }),
    });
    const data = await res.json();
    // recommendations is now [{movieId, title}, ...]
    setRecommendations(data.recommendations);
    setLoading(false);
  };

  const sendFeedback = async (movieId) => {
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movieId, clicked: true }),
    });
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo-row">
          <span className="logo-icon">🎬</span>
          <span className="logo-text">CineRecc</span>
        </div>
        <p className="tagline">Your AI-powered theatre experience</p>
      </header>

      {/* Search */}
      <div className="search-section" ref={searchRef}>
        <div className="search-wrap">
          <span className="search-icon">🔍</span>
          <input
            className="search-input"
            placeholder="Search for movies to add..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            onFocus={() => searchResults.length && setShowDropdown(true)}
          />
        </div>
        {showDropdown && searchResults.length > 0 && (
          <div className="dropdown">
            {searchResults.map((m) => (
              <div key={m.movieId} className="dropdown-item" onClick={() => selectMovie(m)}>
                <span className="dropdown-icon">🎞</span>
                {m.title}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected picks */}
      {selected.length > 0 && (
        <section className="picks-section">
          <h3 className="section-label">🎟 Your Picks</h3>
          <div className="picks-row">
            {selected.map((m) => (
              <div key={m.movieId} className="pick-chip">
                {m.title}
                <button className="chip-remove" onClick={() => removeMovie(m.movieId)}>✕</button>
              </div>
            ))}
          </div>
          <button
            className="cta-btn"
            onClick={getRecommendations}
            disabled={loading}
          >
            {loading ? <span className="spinner" /> : "🎬 Get Recommendations"}
          </button>
        </section>
      )}

      {/* Recommendations */}
      {(recommendations.length > 0 || loading) && (
        <section className="section">
          <h2 className="section-title">
            <span className="section-accent">🍿</span> Recommended For You
          </h2>
          {!loading && recommendations[0]?.fallback && (
            <div className="fallback-notice">
              ⚠️ Your selected movies weren't in our ratings database — showing popular picks instead.
            </div>
          )}
          {loading ? (
            <div className="loading-grid">
              {[...Array(6)].map((_, i) => <div key={i} className="skeleton-card" />)}
            </div>
          ) : (
            <div className="cards-grid">
              {recommendations.map((movie) => (
                <MovieCard
                  key={movie.movieId}
                  movie={movie}
                  onFeedback={sendFeedback}
                  variant="recommendation"
                />
              ))}
            </div>
          )}
        </section>
      )}

      {/* Popular */}
      <section className="section">
        <h2 className="section-title">
          <span className="section-accent">🔥</span> Popular Right Now
        </h2>
        <div className="cards-grid">
          {popular.map((m) => (
            <MovieCard
              key={m.movieId}
              movie={m}
              onFeedback={sendFeedback}
              variant="popular"
            />
          ))}
        </div>
      </section>
    </div>
  );
}
