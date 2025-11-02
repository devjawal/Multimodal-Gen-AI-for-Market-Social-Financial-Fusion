# app.py (Corrected & Enhanced Hybrid Multimodal Pipeline)
import os
import shutil
import joblib
import re
import traceback
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from threading import Lock

# Flask + utils
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import feedparser
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor, early_stopping
import praw
import yfinance as yf
from dateutil import parser
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam

# environment
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import certifi
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception:
    pass

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stock-predictor-v1 by user")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _YT_TRANSCRIPT_AVAILABLE = True
except Exception:
    _YT_TRANSCRIPT_AVAILABLE = False

# Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

# Globals & folders
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
BASE_ART_DIR = os.path.join(os.getcwd(), "artifacts_lgbm_advanced")
STATIC_IMG_DIR = os.path.join(os.getcwd(), "static", "img")
os.makedirs(BASE_ART_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)
_lock = Lock()

# NLP models container
_nlp = {"ok": False, "fin_pipe": None, "embedder": None}
EMB_DIM = 384  # dimension of embedder used (all-MiniLM-L6-v2 -> 384)
EMB_DIM_REDUCED = 16  # number of SVD components per source (tune)

# ---------------------------------------------------------------------------
# Dataclass & helpers
# ---------------------------------------------------------------------------
@dataclass
class Doc:
    date: pd.Timestamp
    text: str
    source: str

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())

def init_nlp():
    """Load finbert pipeline and embedder once per process."""
    if _nlp["ok"]:
        return
    try:
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        _nlp["fin_pipe"] = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512)
        _nlp["embedder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _nlp["ok"] = True
        print("NLP models loaded.")
    except Exception as e:
        print(f"WARNING: NLP models not loaded: {e}")
        _nlp["ok"] = False

def finbert_sentiment_vectors(texts: List[str]) -> np.ndarray:
    """Return Nx3 array [neg, neu, pos] for each input text (fallback neutral if model unavailable)."""
    if not texts:
        return np.zeros((0, 3))
    if not _nlp["ok"]:
        return np.array([[0.0, 1.0, 0.0]] * len(texts))
    results = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        try:
            preds = _nlp["fin_pipe"](chunk)
        except Exception:
            preds = [{"label": "neutral", "score": 1.0} for _ in chunk]
        for p in preds:
            vec = np.zeros(3)
            label = p.get("label", "neutral").lower()
            score = float(p.get("score", 0.0))
            if "negative" in label:
                vec[0] = score
            elif "positive" in label:
                vec[2] = score
            else:
                vec[1] = score
            results.append(vec)
    return np.array(results)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMB_DIM))
    if not _nlp["ok"]:
        return np.zeros((len(texts), EMB_DIM))
    # normalize embeddings to unit vectors
    return _nlp["embedder"].encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

# ---------------------------------------------------------------------------
# Data fetching functions (as in your original code)
# ---------------------------------------------------------------------------
def get_stock_data(ticker: str, days: int) -> pd.DataFrame:
    end, start = datetime.now(), datetime.now() - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No stock data for {ticker}")
    df = df.reset_index()
    # normalize column names into snake_case
    df.columns = [("_".join(filter(None, col)).lower().replace(" ", "_") if isinstance(col, tuple) else col.lower().replace(" ", "_")) for col in df.columns]
    # rename common patterns
    rename_map = {c: c for c in df.columns}
    possible_close = [c for c in df.columns if 'close' in c]
    if not possible_close:
        raise RuntimeError(f"'close' column missing! Columns: {df.columns.tolist()}")
    date_col = next((c for c in df.columns if 'date' in c), None)
    if date_col:
        df.rename(columns={date_col: 'date'}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # ensure standard names exist: open, high, low, close, volume
    # attempt to find matches
    for std in ["open", "high", "low", "close", "volume"]:
        if std not in df.columns:
            match = next((c for c in df.columns if std in c), None)
            if match:
                df.rename(columns={match: std}, inplace=True)
    return df.sort_values("date").reset_index(drop=True)

def fetch_news_rss(queries: list, days: int) -> List[Doc]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    for query in queries:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(f'{query} stock')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries[:300]:
            published = getattr(entry, "published", getattr(entry, "updated", None))
            if not published:
                continue
            try:
                dt = parser.parse(published).astimezone(timezone.utc)
                if dt < cutoff:
                    continue
                text = clean_text(f"{getattr(entry, 'title', '')}. {getattr(entry, 'summary', '')}")
                key = (text[:80], dt.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(dt).normalize(), text=text, source="news"))
                    seen.add(key)
            except Exception:
                continue
    return docs

def fetch_reddit_praw(queries: list, days: int) -> List[Doc]:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        return []
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    subs = "wallstreetbets+stocks+investing+technology+business"
    for query in queries:
        try:
            for submission in reddit.subreddit(subs).search(query, sort="new", limit=400):
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                if created < cutoff:
                    continue
                text = clean_text(f"{submission.title}. {submission.selftext or ''}")
                key = (submission.id, created.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(created).normalize(), text=text, source="reddit"))
                    seen.add(key)
        except Exception as e:
            print(f"WARNING: reddit fetch error: {e}")
    return docs

def fetch_youtube_docs(queries: list, days: int) -> List[Doc]:
    if not YOUTUBE_API_KEY:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    for query in queries:
        params = {"part": "snippet", "q": f'"{query}" stock analysis', "type": "video", "order": "date", "maxResults": 50, "key": YOUTUBE_API_KEY, "publishedAfter": cutoff.isoformat("T").replace("+00:00", "Z")}
        try:
            r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=20)
            r.raise_for_status()
            for item in r.json().get("items", []):
                video_id = item["id"].get("videoId")
                if not video_id or video_id in seen:
                    continue
                published_at = parser.parse(item["snippet"]["publishedAt"])
                title = item["snippet"]["title"]
                transcript = ""
                if _YT_TRANSCRIPT_AVAILABLE:
                    try:
                        transcript = " ".join([t["text"] for t in YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])])
                    except Exception:
                        pass
                text = clean_text(f"{title}. {transcript}")
                if text:
                    docs.append(Doc(date=pd.to_datetime(published_at).normalize(), text=text, source="youtube"))
                    seen.add(video_id)
        except Exception as e:
            print(f"WARNING: YT fetch failed for '{query}': {e}")
    return docs

# ---------------------------------------------------------------------------
# Aggregation, detection and feature engineering
# ---------------------------------------------------------------------------
def aggregate_text_features(docs: List[Doc], source: str) -> pd.DataFrame:
    """
    Takes a list of Doc and returns a daily aggregated DataFrame with:
    date + averaged sentiment (neg/neu/pos) + averaged embedding components (EMB_DIM columns).
    """
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": d.date, "text": d.text} for d in docs])
    df["text"] = df["text"].astype(str)
    # compute finbert sentiments
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=[f"{source}_sent_neg", f"{source}_sent_neu", f"{source}_sent_pos"])
    # compute embeddings
    embeddings = embed_texts(df["text"].tolist())
    emb_df = pd.DataFrame(embeddings, columns=[f"{source}_emb_{i}" for i in range(embeddings.shape[1])])
    # combine and group by date
    full = pd.concat([df.drop(columns="text").reset_index(drop=True), sent_df, emb_df], axis=1)
    daily = full.groupby("date").mean().reset_index()
    return daily

def detect_event_features(all_docs: List[Doc]) -> pd.DataFrame:
    """
    Detect potentially impactful event days using a keyword dictionary.
    Returns daily df with event sentiment and is_event_day flag.
    """
    if not all_docs:
        return pd.DataFrame()
    EVENT_KEYWORDS = {
        "earnings": ["earnings", "quarterly results", "revenue", "profit", "eps", "guidance"],
        "launch": ["launch", "announcement", "release", "unveiled", "introduc"],
        "legal": ["lawsuit", "investigation", "sec", "doj", "settlement", "charge"],
        "partnership": ["partnership", "collaboration", "acquisition", "merger", "buyout"]
    }
    rows = []
    for doc in all_docs:
        text = doc.text or ""
        found = False
        for keywords in EVENT_KEYWORDS.values():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text, flags=re.IGNORECASE):
                    found = True
                    break
            if found:
                break
        if found:
            rows.append({"date": doc.date, "text": text})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates()
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=["event_sent_neg", "event_sent_neu", "event_sent_pos"])
    full = pd.concat([df.drop(columns="text").reset_index(drop=True), sent_df], axis=1)
    daily = full.groupby("date").mean().reset_index()
    daily["is_event_day"] = 1
    return daily

# ---------------------------------------------------------------------------
# Attention weights and embedding reduction utilities
# ---------------------------------------------------------------------------
def compute_attention_weights_vectorized(vol_array: np.ndarray,
                                         volume_array: np.ndarray,
                                         is_event_array: np.ndarray,
                                         vol_pow: float = 1.0,
                                         volume_pow: float = 1.0,
                                         event_boost: float = 3.0,
                                         eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized computation of attention weights for each day.
    Returns array shape (n_days, 4) with normalized weights for [news, reddit, yt, event].
    Heuristic: news driven by volatility, reddit/yt driven by volume, event gets boost.
    """
    vol = np.maximum(0.0, np.nan_to_num(vol_array, nan=0.0)) ** vol_pow
    volu = np.maximum(0.0, np.nan_to_num(volume_array, nan=0.0)) ** volume_pow
    is_evt = np.where(np.nan_to_num(is_event_array, nan=0.0) >= 1.0, 1.0, 0.0)

    # raw weights
    raw_news = vol + 0.1
    raw_reddit = volu + 0.05
    raw_yt = volu * 0.7 + 0.05
    raw_event = is_evt * event_boost

    # amplify when event present
    raw_news = raw_news * (1.0 + 0.4 * is_evt)
    raw_reddit = raw_reddit * (1.0 + 0.2 * is_evt)
    raw_yt = raw_yt * (1.0 + 0.2 * is_evt)
    raw_event = raw_event + 0.01  # ensure small positive

    raw = np.vstack([raw_news, raw_reddit, raw_yt, raw_event]).T  # shape (n,4)
    denom = raw.sum(axis=1, keepdims=True)
    denom = np.where(denom <= eps, 1.0, denom)
    weights = raw / denom
    return weights  # columns correspond to [w_news, w_reddit, w_yt, w_event]

def reduce_embedding_matrix(emb_df: pd.DataFrame, n_components: int = EMB_DIM_REDUCED, reducer: TruncatedSVD = None):
    """
    Reduce high-dim embedding DF to n_components using TruncatedSVD.
    If reducer is None -> fit new reducer and return (reduced_df, reducer).
    Else -> transform and return (reduced_df, reducer).
    """
    if emb_df.shape[0] == 0:
        return pd.DataFrame(), reducer
    # keep original index for merge
    idx = emb_df['date'] if 'date' in emb_df.columns else emb_df.index
    emb_only = emb_df.drop(columns=['date'], errors='ignore').fillna(0.0).values
    if reducer is None:
        reducer = TruncatedSVD(n_components=min(n_components, emb_only.shape[1]-1 if emb_only.shape[1]>1 else 1), random_state=RANDOM_SEED)
        transformed = reducer.fit_transform(emb_only)
    else:
        transformed = reducer.transform(emb_only)
    cols = [f"emb_svd_{i}" for i in range(transformed.shape[1])]
    reduced_df = pd.DataFrame(transformed, columns=cols, index=range(len(transformed)))
    # attach date column back if present in original
    if 'date' in emb_df.columns:
        reduced_df.insert(0, 'date', pd.to_datetime(emb_df['date']).dt.tz_localize(None))
    return reduced_df.reset_index(drop=True), reducer

# ---------------------------------------------------------------------------
# Hybrid feature builder (now supports embedding reducers passed in)
# ---------------------------------------------------------------------------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators used in the pipeline:
    - simple returns (1, 5)
    - SMA for windows [5,10,20,60,120]
    - RSI using rolling average of gains/losses per window
    - rolling volatility (std of ret_1) for windows
    Returns a new DataFrame (copy) with added columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # Ensure close column exists
    if "close" not in out.columns:
        raise RuntimeError("'close' column required in DataFrame for technical features")

    close = out["close"].astype(float)

    # basic returns
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)

    windows = [5, 10, 20, 60, 120]
    # compute SMA, RSI (Wilder-like via simple rolling mean of gains/losses), and vol
    delta = close.diff()

    for win in windows:
        # SMA
        out[f"sma_{win}"] = close.rolling(window=win, min_periods=1).mean()

        # Gains / Losses for RSI-like metric (simple rolling mean)
        gain = delta.where(delta > 0, 0.0).rolling(window=win, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(window=win, min_periods=1).mean()

        # Avoid division by zero
        rs = gain / (loss + 1e-9)
        out[f"rsi_{win}"] = 100.0 - (100.0 / (1.0 + rs))

        # volatility of 1-day returns over window
        out[f"vol_{win}"] = out["ret_1"].rolling(window=win, min_periods=1).std()

    # fill NaNs reasonably: keep NaNs where appropriate but avoid infinite values
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Note: do not drop rows here; caller will decide how to handle NaNs
    return out


def build_hybrid_featureset(stock_df: pd.DataFrame,
                            news_df: pd.DataFrame,
                            reddit_df: pd.DataFrame,
                            yt_df: pd.DataFrame,
                            event_df: pd.DataFrame,
                            emb_reducers: Dict[str, TruncatedSVD] = None) -> pd.DataFrame:
    """
    Merge market data and daily text features.
    emb_reducers: optional mapping like {'news': svd_news, 'reddit': svd_reddit, 'yt': svd_yt}
    If reducers provided, the function will transform embedding columns into reduced components before lagging.
    """
    features = stock_df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)
    # merge each text df
    for text_df, src in [(news_df, "news"), (reddit_df, "reddit"), (yt_df, "yt"), (event_df, "event")]:
        if text_df is not None and not text_df.empty:
            text_df_loc = text_df.copy()
            text_df_loc["date"] = pd.to_datetime(text_df_loc["date"]).dt.tz_localize(None)
            features = pd.merge(features, text_df_loc, on="date", how="left")
    # technicals
    features = add_technical_features(features)
    # scaled versions for vol and volume
    scaler_vol = MinMaxScaler()
    features['vol_20_scaled'] = scaler_vol.fit_transform(features[['vol_20']].fillna(0))
    scaler_volu = MinMaxScaler()
    features['volume_scaled'] = scaler_volu.fit_transform(features[['volume']].fillna(0))

    # Ensure event flag exists
    features['is_event_day'] = features.get('is_event_day', 0).fillna(0)

    # Prepare reduced embeddings per source if reducers provided
    reduced_dfs = {}
    for src in ['news', 'reddit', 'yt']:
        emb_cols = [c for c in features.columns if c.startswith(f"{src}_emb_")]
        if emb_cols:
            emb_df = features[['date'] + emb_cols].copy()
            if emb_reducers and src in emb_reducers and emb_reducers[src] is not None:
                reduced, _ = reduce_embedding_matrix(emb_df, n_components=EMB_DIM_REDUCED, reducer=emb_reducers[src])
            else:
                reduced, _ = reduce_embedding_matrix(emb_df, n_components=EMB_DIM_REDUCED, reducer=None)
            # merge reduced back by date
            features = features.merge(reduced, on='date', how='left')
            # reduced may include a 'date' column for the merge; don't treat 'date' as an emb feature
            reduced_cols = [c for c in reduced.columns.tolist() if c != 'date']
            reduced_dfs[src] = reduced_cols

    # compute per-source sentiment scores
    features['news_sentiment_score'] = features.get('news_sent_pos', 0) - features.get('news_sent_neg', 0)
    features['reddit_sentiment_score'] = features.get('reddit_sent_pos', 0) - features.get('reddit_sent_neg', 0)
    features['yt_sentiment_score'] = features.get('yt_sent_pos', 0) - features.get('yt_sent_neg', 0)
    features['event_sentiment_score'] = features.get('event_sent_pos', 0) - features.get('event_sent_neg', 0)

    # compute normalized attention weights vectorized
    vol_arr = features['vol_20'].fillna(0).values
    volu_arr = features['volume'].fillna(0).values
    is_evt_arr = features['is_event_day'].fillna(0).values
    weights = compute_attention_weights_vectorized(vol_arr, volu_arr, is_evt_arr)
    # attach weights as columns
    features[['w_news', 'w_reddit', 'w_yt', 'w_event']] = pd.DataFrame(weights, index=features.index)

    # combined signal
    features['combined_sentiment_signal'] = (
        features['w_news'] * features['news_sentiment_score'].fillna(0) +
        features['w_reddit'] * features['reddit_sentiment_score'].fillna(0) +
        features['w_yt'] * features['yt_sentiment_score'].fillna(0) +
        features['w_event'] * features['event_sentiment_score'].fillna(0)
    )

    # Now create lag features — select a subset to lag to reduce explosion:
    # choose: key technicals + combined_sentiment_signal + reduced embedding components
    lag_periods = [1, 2, 3, 5, 10, 21, 63]
    base_cols_to_lag = ['ret_1', 'ret_5', 'sma_5', 'sma_10', 'sma_20', 'sma_60', 'rsi_20', 'vol_20', 'combined_sentiment_signal']
    # include reduced embedding columns if present:
    for src in reduced_dfs:
        base_cols_to_lag += reduced_dfs[src]
    base_cols_to_lag = [c for c in base_cols_to_lag if c in features.columns]

    for lag in lag_periods:
        shifted = features[base_cols_to_lag].shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        features = pd.concat([features, shifted], axis=1)

    # target: next-day close
    features['target'] = features['close'].shift(-1)
    # drop rows with missing target
    features = features.dropna(subset=['target']).fillna(0).reset_index(drop=True)
    return features, reduced_dfs

# ---------------------------------------------------------------------------
# Residual LSTM utilities (train on train residuals; iterative predict for test)
# ---------------------------------------------------------------------------
def build_residual_sequences_from_series(residuals_series: pd.Series, lookback=10):
    residuals = residuals_series.values
    X, y = [], []
    for i in range(len(residuals) - lookback):
        X.append(residuals[i:i+lookback])
        y.append(residuals[i+lookback])
    if len(X) == 0:
        return np.empty((0, lookback)), np.empty((0,))
    return np.array(X), np.array(y)

def train_residual_lstm_from_residuals(residuals_series: pd.Series, lookback=10, epochs=20):
    X, y = build_residual_sequences_from_series(residuals_series, lookback)
    if len(X) < 2:
        return None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    return model

def predict_residuals_iterative(residual_lstm, resid_seed: np.ndarray, n_preds: int, lookback=10):
    """
    resid_seed: 1D array with at least 'lookback' last residuals (typically last lookback residuals from train)
    Predict n_preds residuals iteratively, appending each predicted residual and sliding window forward.
    """
    if residual_lstm is None or len(resid_seed) < lookback:
        return np.zeros(n_preds)
    seed = list(resid_seed[-lookback:].tolist())
    preds = []
    for _ in range(n_preds):
        X_in = np.array(seed[-lookback:]).reshape((1, lookback, 1))
        pred = float(residual_lstm.predict(X_in, verbose=0).flatten()[0])
        preds.append(pred)
        seed.append(pred)
    return np.array(preds)

# ---------------------------------------------------------------------------
# Model training & prediction wrappers
# ---------------------------------------------------------------------------
def generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, query, predicted_price=None):
    if not GEMINI_API_KEY:
        return "Gemini API key not found."
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        texts = [f"[{d.source}] {d.text}" for d in (news_docs + reddit_docs + yt_docs)[:45]]
        extra = f"\nThe model predicts the next closing price will be around ${predicted_price:.2f}." if predicted_price else ""
        full_text = " ".join(texts)
        truncated_text = full_text[:10000] + ('...' if len(full_text) > 10000 else '')
        prompt = f"Summarize these financial signals for {ticker}, focusing on key events, sentiment, and conflicts.\n\nRecent Texts:\n{truncated_text}\n{extra}"
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Summary generation failed: {e}"

def get_ticker_paths(ticker: str):
    safe_ticker = ticker.upper().replace("/", "-")
    art_dir = os.path.join(BASE_ART_DIR, safe_ticker)
    plot_dir = os.path.join(STATIC_IMG_DIR, safe_ticker)
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "lgbm": os.path.join(art_dir, "lgbm_model.pkl"),
        "scaler": os.path.join(art_dir, "scaler.pkl"),
        "meta": os.path.join(art_dir, "pipeline_meta.pkl"),
        "lstm": os.path.join(art_dir, "residual_lstm.keras"),
        "summary": os.path.join(art_dir, "summary.txt"),
        "plot": os.path.join(plot_dir, "prediction_comparison.png"),
        "plot_url": f"/static/img/{safe_ticker}/prediction_comparison.png"
    }

def compute_advanced_signal(last_feature_row: pd.Series, last_pred: float) -> str:
    last_close = last_feature_row.get('close', 0)
    if last_close == 0:
        return "HOLD"
    price_pct_change = (last_pred - last_close) / last_close
    rsi = last_feature_row.get('rsi_20', 50)
    sma_20 = last_feature_row.get('sma_20', last_close)
    sma_60 = last_feature_row.get('sma_60', last_close)
    in_uptrend = sma_20 > sma_60
    news_att = last_feature_row.get('news_attention_score', 0)
    reddit_att = last_feature_row.get('reddit_attention_score', 0)
    event_sent = last_feature_row.get('event_sent_pos', 0) - last_feature_row.get('event_sent_neg', 0)
    total_sentiment = news_att + reddit_att + event_sent

    if price_pct_change > 0.02:
        return "STRONG BUY"
    if price_pct_change < -0.02:
        return "STRONG SELL"
    if price_pct_change > 0.005:
        if in_uptrend and total_sentiment > 0.05:
            return "BUY"
        if total_sentiment < -0.1 or rsi > 75:
            return "HOLD"
        return "BUY"
    if price_pct_change < -0.005:
        if not in_uptrend and total_sentiment < -0.05:
            return "SELL"
        if total_sentiment > 0.1 or rsi < 25:
            return "HOLD"
        return "SELL"
    return "HOLD"

def run_full_training(ticker: str, queries: list, days: int):
    """Main pipeline that trains LGBM on train set, trains residual LSTM on train residuals,
       iteratively predicts residuals on test and combines them into final predictions."""
    paths = get_ticker_paths(ticker)
    init_nlp()
    stock_df = get_stock_data(ticker, days=days)
    news_docs = fetch_news_rss(queries, days=days)
    reddit_docs = fetch_reddit_praw(queries, days=days)
    yt_docs = fetch_youtube_docs(queries, days=days)
    all_docs = news_docs + reddit_docs + yt_docs

    news_daily = aggregate_text_features(news_docs, "news")
    reddit_daily = aggregate_text_features(reddit_docs, "reddit")
    yt_daily = aggregate_text_features(yt_docs, "yt")
    event_daily = detect_event_features(all_docs)

    # Step 1: Merge all text and stock data, but DO NOT create embeddings or lags yet.
    features = stock_df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)
    for text_df in [news_daily, reddit_daily, yt_daily, event_daily]:
        if text_df is not None and not text_df.empty:
            text_df_loc = text_df.copy()
            text_df_loc["date"] = pd.to_datetime(text_df_loc["date"]).dt.tz_localize(None)
            features = pd.merge(features, text_df_loc, on="date", how="left")

    # Step 2: Split data to find the training set for fitting SVD reducers
    split_index = int(len(features) * 0.8)
    train_df_for_svd = features.iloc[:split_index]

    # Step 3: Fit SVD reducers ONLY on the training portion of the embedding columns
    emb_reducers = {}
    for src in ['news', 'reddit', 'yt']:
        emb_cols = [c for c in train_df_for_svd.columns if c.startswith(f"{src}_emb_")]
        if not emb_cols:
            continue
        emb_train_df = train_df_for_svd[['date'] + emb_cols].copy().dropna()
        if not emb_train_df.empty:
            _, reducer = reduce_embedding_matrix(
                emb_train_df,
                n_components=min(EMB_DIM_REDUCED, len(emb_cols))
            )
            emb_reducers[src] = reducer

    # Step 4: Now, build the complete, final feature set ONCE
    # Add technicals, reduced embeddings, sentiment signals, and lags in one go.
    
    # Add technicals
    features = add_technical_features(features)
    
    # Add reduced embeddings using the fitted reducers
    reduced_dfs_cols = {}
    for src in ['news', 'reddit', 'yt']:
        emb_cols = [c for c in features.columns if c.startswith(f"{src}_emb_")]
        if emb_cols and src in emb_reducers:
            emb_df = features[['date'] + emb_cols].copy()
            reduced, _ = reduce_embedding_matrix(emb_df, reducer=emb_reducers[src])
            # Add a source prefix to avoid column name collisions before merging
            reduced.columns = [f"{src}_{col}" if col != 'date' else 'date' for col in reduced.columns]
            features = features.merge(reduced, on='date', how='left')
            reduced_dfs_cols[src] = [c for c in reduced.columns if c != 'date']

    # Add attention and combined sentiment signals
    features['vol_20_scaled'] = MinMaxScaler().fit_transform(features[['vol_20']].fillna(0))
    features['volume_scaled'] = MinMaxScaler().fit_transform(features[['volume']].fillna(0))
    features['is_event_day'] = features.get('is_event_day', 0).fillna(0)
    weights = compute_attention_weights_vectorized(
        features['vol_20'].fillna(0).values,
        features['volume'].fillna(0).values,
        features['is_event_day'].fillna(0).values
    )
    features[['w_news', 'w_reddit', 'w_yt', 'w_event']] = pd.DataFrame(weights, index=features.index)
    features['news_sentiment_score'] = features.get('news_sent_pos', 0) - features.get('news_sent_neg', 0)
    features['reddit_sentiment_score'] = features.get('reddit_sent_pos', 0) - features.get('reddit_sent_neg', 0)
    features['yt_sentiment_score'] = features.get('yt_sent_pos', 0) - features.get('yt_sent_neg', 0)
    features['event_sentiment_score'] = features.get('event_sent_pos', 0) - features.get('event_sent_neg', 0)
    features['combined_sentiment_signal'] = (
        features['w_news'] * features['news_sentiment_score'].fillna(0) +
        features['w_reddit'] * features['reddit_sentiment_score'].fillna(0) +
        features['w_yt'] * features['yt_sentiment_score'].fillna(0) +
        features['w_event'] * features['event_sentiment_score'].fillna(0)
    )

    # Add lag features
    lag_periods = [1, 2, 3, 5, 10, 21, 63]
    base_cols_to_lag = ['ret_1', 'ret_5', 'sma_5', 'sma_10', 'sma_20', 'sma_60', 'rsi_20', 'vol_20', 'combined_sentiment_signal']
    for src in reduced_dfs_cols:
        base_cols_to_lag += reduced_dfs_cols[src]
    
    # Ensure all columns to be lagged actually exist
    base_cols_to_lag = [c for c in base_cols_to_lag if c in features.columns]

    for lag in lag_periods:
        shifted = features[base_cols_to_lag].shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        features = pd.concat([features, shifted], axis=1)

    # Set target and clean up
    features['target'] = features['close'].shift(-1)
    features_full_reduced = features.dropna(subset=['target']).fillna(0).reset_index(drop=True)

    if features_full_reduced.empty:
        raise RuntimeError("No data after final feature engineering")

    # Step 5: Split the final featureset and train the model
    train_features = features_full_reduced.iloc[:split_index].reset_index(drop=True)
    test_features = features_full_reduced.iloc[split_index:].reset_index(drop=True)

    # Define feature columns to use (exclude raw embeddings, identifiers, etc.)
    feature_cols = [c for c in train_features.columns if not (c.startswith(('news_emb_', 'reddit_emb_', 'yt_emb_')) or c in ['date', 'target', 'close'])]
    
    # Drop duplicates just in case
    feature_cols = sorted(list(set(feature_cols)))

    X_train = train_features[feature_cols]
    y_train = train_features['target']
    X_test = test_features[feature_cols]
    y_test = test_features['target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=RANDOM_SEED)
    model.fit(X_train_scaled_df, y_train, eval_set=[(X_test_scaled_df, y_test)], eval_metric='l1', callbacks=[early_stopping(100, verbose=False)])
    
    # --- The rest of your function (predictions, LSTM, saving, etc.) remains the same ---
    preds_train = model.predict(X_train_scaled_df)
    preds_test = model.predict(X_test_scaled_df)

    # residuals on train -> train LSTM on train residuals
    residuals_train = y_train.reset_index(drop=True) - pd.Series(preds_train)
    lookback = 10
    residual_lstm = train_residual_lstm_from_residuals(residuals_train, lookback=lookback, epochs=20)

    corrected_preds = preds_test.copy()
    if residual_lstm is not None and len(test_features) > 0:
        seed_resid = residuals_train.values[-lookback:] if len(residuals_train) >= lookback else np.pad(residuals_train.values, (lookback - len(residuals_train), 0))
        resid_preds_test = predict_residuals_iterative(residual_lstm, seed_resid, len(preds_test), lookback=lookback)
        corrected_preds = preds_test + resid_preds_test
        try:
            residual_lstm.save(paths["lstm"])
        except Exception as e:
            print(f"Warning saving LSTM: {e}")

    final_mae = mean_absolute_error(y_test, corrected_preds) if len(y_test) > 0 else None
    final_r2 = r2_score(y_test, corrected_preds) if len(y_test) > 0 else None

    # save models & meta
    joblib.dump(model, paths["lgbm"])
    joblib.dump(scaler, paths["scaler"])
    meta = {
        "feature_names": list(X_train.columns),
        "lookback": lookback,
        "mae_train": float(final_mae) if final_mae is not None else None,
        "r2_train": float(final_r2) if final_r2 is not None else None,
        "training_date": datetime.utcnow().isoformat(),
        "emb_reducers_present": list(emb_reducers.keys())
    }
    joblib.dump(meta, paths["meta"])

    # plotting + summary
    if len(test_features) > 0:
        last_pred = float(corrected_preds[-1])
        summary = generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, queries, last_pred)
        with open(paths["summary"], "w", encoding='utf-8') as f:
            f.write(summary)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(test_features['date'], y_test, label="Actual Price", color="royalblue")
        ax.plot(test_features['date'], preds_test, label="LGBM Prediction", color="darkorange", ls="--")
        if residual_lstm is not None:
            ax.plot(test_features['date'], corrected_preds, label="Hybrid (LGBM + LSTM)", color="green", ls="-.")
        ax.set_title(f"{ticker} Price Prediction", fontsize=16)
        ax.legend()
        fig.tight_layout()
        plt.savefig(paths["plot"])
        plt.close(fig)
    else:
        last_pred = float(preds_train[-1]) if len(preds_train) > 0 else 0.0
        summary = generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, queries, last_pred)
        with open(paths["summary"], "w", encoding='utf-8') as f:
            f.write(summary)

    signal = compute_advanced_signal(test_features.iloc[-1] if len(test_features) > 0 else train_features.iloc[-1] if len(train_features)>0 else pd.Series(), last_pred)
    return {"last_pred": float(last_pred), "signal": signal}

def predict_fast(ticker: str):
    paths = get_ticker_paths(ticker)
    # load artifacts
    model = joblib.load(paths["lgbm"])
    scaler = joblib.load(paths["scaler"])
    meta = joblib.load(paths["meta"])
    feature_names = meta.get("feature_names", [])

    days_to_fetch = 120 + 63 + 10
    stock_df = get_stock_data(ticker, days_to_fetch)
    # fast features without text (or can fetch a small window of texts and reduce)
    features, _ = build_hybrid_featureset(stock_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), emb_reducers=None)
    if features.empty:
        return {"last_pred": 0.0, "signal": "HOLD"}
    last_row = features.tail(1)
    X_predict = last_row.drop(columns=['date', 'target', 'close'], errors='ignore').reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X_predict)
    prediction = model.predict(X_scaled)[0]
    signal = compute_advanced_signal(last_row.iloc[0], prediction)
    return {"last_pred": float(prediction), "signal": signal}

# ---------------------------------------------------------------------------
# Flask routes (unchanged mostly)
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index_route():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if not _lock.acquire(blocking=False):
        flash("Another process is already running. Please wait.", "error")
        return redirect(url_for("index_route"))
    try:
        ticker = request.form.get("ticker", "").strip().upper()
        queries = [q.strip() for q in request.form.get("queries", "").split(",")] if request.form.get("queries") else [ticker]
        days = int(request.form.get("days", 1200))
        retrain = "retrain" in request.form

        if not ticker:
            flash("Ticker symbol is required.", "error")
            return redirect(url_for("index_route"))

        paths = get_ticker_paths(ticker)
        artifacts_exist = os.path.exists(paths["lgbm"])

        if retrain or not artifacts_exist:
            flash(f"Training new model for {ticker}. This will take several minutes...", "info")
            results = run_full_training(ticker, queries, days)
        else:
            flash(f"Using existing model for a fast prediction for {ticker}.", "info")
            results = predict_fast(ticker)
        meta = joblib.load(paths["meta"]) if os.path.exists(paths["meta"]) else {}
        summary = ""
        if os.path.exists(paths["summary"]):
            with open(paths["summary"], 'r', encoding='utf-8') as f:
                summary = f.read()
        return render_template("result.html", ticker=ticker, results=results, meta=meta, summary=summary, plot_url=paths["plot_url"])
    except Exception as e:
        traceback.print_exc()
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for("index_route"))
    finally:
        _lock.release()

# ---------------------------------------------------------------------------
# Create small default templates + run (same as before)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    # simple templates (same as your earlier app)
    with open("templates/layout.html", "w") as f:
        f.write("""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Multimodal Stock Predictor</title><link rel="stylesheet" href="/static/style.css"></head><body><div class="container"><header><h1>📈 Multimodal Stock Predictor</h1></header><main>{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}{% for category, message in messages %}<div class="flash {{ category }}">{{ message }}</div>{% endfor %}{% endif %}{% endwith %}{% block content %}{% endblock %}</main><footer><p>Powered by LightGBM, LSTM, and Multimodal Data Fusion</p></footer></div></body></html>""")
    with open("templates/index.html", "w") as f:
        f.write("""{% extends "layout.html" %}{% block content %}<div class="card"><form method="post" action="/predict"><div class="form-group"><label for="ticker">Ticker Symbol</label><input id="ticker" name="ticker" value="AAPL" required placeholder="e.g., AAPL, GOOG"></div><div class="form-group"><label for="queries">Search Queries (comma-separated)</label><input id="queries" name="queries" value="Apple, Iphone, Tim cook"></div><div class="form-group"><label for="days">Training History (Days)</label><input id="days" name="days" type="number" value="1200"></div><div class="form-group checkbox-group"><input type="checkbox" id="retrain" name="retrain"><label for="retrain">Force Full Retraining</label></div><button type="submit">Analyze & Predict</button></form></div>{% endblock %}""")
    with open("templates/result.html", "w") as f:
        f.write("""{% extends "layout.html" %}{% block content %}<div class="card result-card"><h2>Results for {{ ticker }}</h2><div class="signal-box">Prediction: <strong>${{ "%.2f"|format(results.last_pred) }}</strong><span class="signal {{ results.signal|lower|replace(' ', '-') }}">{{ results.signal }}</span></div>{% if meta.mae_train %}<div class="metrics"><h4>Last Full Training Performance</h4><p><strong>MAE:</strong> {{ "%.4f"|format(meta.mae_train) }}</p><p><strong>R² Score:</strong> {{ "%.4f"|format(meta.r2_train) }}</p><p class="muted">Trained on: {{ meta.training_date.split('T')[0] }}</p></div>{% endif %}{% if plot_url and meta.mae_train %}<div class="plot"><h3>Prediction Chart</h3><img src="{{ plot_url }}?t={{ range(1,100000) | random }}" alt="Prediction plot"></div>{% endif %}<div class="summary"><h3>AI-Generated Summary</h3><pre>{{ summary or 'Summary not available.' }}</pre></div><a href="/" class="back-link">← Make another prediction</a></div>{% endblock %}""")
    # minimal CSS
    with open("static/style.css", "w") as f:
        f.write("body{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;padding:20px}.container{max-width:900px;margin:0 auto}.card{background:#fff;padding:20px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06)}")
    app.run(host="0.0.0.0", port=5691, debug=True)
