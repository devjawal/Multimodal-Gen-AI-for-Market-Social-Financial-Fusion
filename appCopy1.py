# app.py (Final Version with Corrected Advanced Recommendation Logic)

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

# --- All Necessary Imports ---
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for web servers
import matplotlib.pyplot as plt
import requests
import feedparser
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, early_stopping
import praw
import yfinance as yf
from dateutil import parser
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam # Use legacy optimizer for M1/M2 Mac compatibility

# --- Environment & Setup ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import certifi
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
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

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "a-very-secure-secret-key-change-me")

# --- Directory & Global Config ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
BASE_ART_DIR = os.path.join(os.getcwd(), "artifacts_lgbm_advanced")
STATIC_IMG_DIR = os.path.join(os.getcwd(), "static", "img")
os.makedirs(BASE_ART_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)
_lock = Lock()

#
# ==============================================================================
#  SECTION 1: YOUR COMPLETE after1.py PIPELINE (UNCHANGED)
# ==============================================================================
#
@dataclass
class Doc:
    date: pd.Timestamp
    text: str
    source: str

def clean_text(text: str) -> str:
    if not text: return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())

_nlp = {"ok": False, "fin_pipe": None, "embedder": None}
EMB_DIM = 384

def init_nlp():
    if _nlp["ok"]: return
    print("Loading NLP models...")
    try:
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        _nlp["fin_pipe"] = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512)
        _nlp["embedder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _nlp["ok"] = True
        print("NLP models loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load NLP models. Error: {repr(e)}")

def finbert_sentiment_vectors(texts: List[str]) -> np.ndarray:
    if not texts or not _nlp["ok"]: return np.array([[0.0, 1.0, 0.0]] * len(texts))
    results = []
    for text_chunk in [texts[i:i + 16] for i in range(0, len(texts), 16)]:
        try:
            preds = _nlp["fin_pipe"](text_chunk)
            for p in preds:
                vec = np.zeros(3)
                label, score = p.get("label", "neutral").lower(), p.get("score", 0.0)
                if "negative" in label: vec[0] = score
                elif "positive" in label: vec[2] = score
                else: vec[1] = score
                results.append(vec)
        except Exception: results.extend([np.array([0.0, 1.0, 0.0])] * len(text_chunk))
    return np.array(results)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts or not _nlp["ok"]: return np.zeros((len(texts), EMB_DIM))
    return _nlp["embedder"].encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)

def get_stock_data(ticker: str, days: int) -> pd.DataFrame:
    end, start = datetime.now(), datetime.now() - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: raise RuntimeError(f"No stock data for {ticker}")
    df = df.reset_index()
    df.columns = ["_".join(filter(None, col)).lower().replace(" ", "_") if isinstance(col, tuple) else col.lower().replace(" ", "_") for col in df.columns]
    rename_map = {"adj_close": "close", f"close_{ticker.lower()}": "close", f"open_{ticker.lower()}": "open", f"high_{ticker.lower()}": "high", f"low_{ticker.lower()}": "low", f"volume_{ticker.lower()}": "volume"}
    df = df.rename(columns=rename_map)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if not date_col: raise RuntimeError(f"'date' column missing! Columns: {df.columns.tolist()}")
    df.rename(columns={date_col: 'date'}, inplace=True)
    if "close" not in df.columns: raise RuntimeError(f"'close' column missing! Columns: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)

def fetch_news_rss(queries: list, days: int) -> List[Doc]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    for query in queries:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(f'{query} stock')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for entry in feed.entries[:200]:
            published = getattr(entry, "published", getattr(entry, "updated", None))
            if not published: continue
            try:
                dt = parser.parse(published).astimezone(timezone.utc)
                if dt < cutoff: continue
                text = clean_text(f"{getattr(entry, 'title', '')}. {getattr(entry, 'summary', '')}")
                key = (text[:50], dt.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(dt).normalize(), text=text, source="news"))
                    seen.add(key)
            except Exception: continue
    return docs

def fetch_reddit_praw(queries: list, days: int) -> List[Doc]:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]): return []
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    try:
        subs = "wallstreetbets+stocks+investing+apple+technology"
        for query in queries:
            for submission in reddit.subreddit(subs).search(query, sort="new", limit=300):
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                if created < cutoff: continue
                text = clean_text(f"{submission.title}. {submission.selftext or ''}")
                key = (submission.id, created.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(created).normalize(), text=text, source="reddit"))
                    seen.add(key)
    except Exception as e: print(f"WARNING: Reddit fetch failed: {e}")
    return docs

def fetch_youtube_docs(queries: list, days: int) -> List[Doc]:
    if not YOUTUBE_API_KEY: return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    for query in queries:
        params = {"part": "snippet", "q": f'"{query}" stock analysis', "type": "video", "order": "date", "maxResults": 50, "key": YOUTUBE_API_KEY, "publishedAfter": cutoff.isoformat("T").replace("+00:00", "Z")}
        try:
            r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=20)
            r.raise_for_status()
            for item in r.json().get("items", []):
                video_id = item["id"]["videoId"]
                if video_id in seen: continue
                published_at = parser.parse(item["snippet"]["publishedAt"])
                title = item["snippet"]["title"]
                transcript = ""
                if _YT_TRANSCRIPT_AVAILABLE:
                    try:
                        transcript = " ".join([t["text"] for t in YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])])
                    except Exception: pass
                text = clean_text(f"{title}. {transcript}")
                if text:
                    docs.append(Doc(date=pd.to_datetime(published_at).normalize(), text=text, source="youtube"))
                    seen.add(video_id)
        except Exception as e: print(f"WARNING: YouTube fetch failed for query '{query}': {e}")
    return docs

def aggregate_text_features(docs: List[Doc], source: str) -> pd.DataFrame:
    if not docs: return pd.DataFrame()
    df = pd.DataFrame([{"date": d.date, "text": d.text} for d in docs])
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=[f"{source}_sent_neg", f"{source}_sent_neu", f"{source}_sent_pos"])
    embeddings = embed_texts(df["text"].tolist())
    emb_df = pd.DataFrame(embeddings, columns=[f"{source}_emb_{i}" for i in range(EMB_DIM)])
    full_df = pd.concat([df.drop(columns="text"), sent_df, emb_df], axis=1)
    return full_df.groupby("date").mean().reset_index()

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out, close = df.copy(), df["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    for win in [5, 10, 20, 60, 120]:
        out[f"sma_{win}"] = close.rolling(win).mean()
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(win).mean()
        loss = -delta.where(delta < 0, 0).rolling(win).mean()
        rs = gain / (loss + 1e-9)
        out[f"rsi_{win}"] = 100 - (100 / (1 + rs))
        out[f"vol_{win}"] = out["ret_1"].rolling(win).std()
    return out

def detect_event_features(all_docs: List[Doc]) -> pd.DataFrame:
    if not all_docs: return pd.DataFrame()
    EVENT_KEYWORDS = {"earnings": ["earnings", "quarterly results", "revenue", "profit"], "launch": ["launch", "announcement", "release", "unveiled"], "legal": ["lawsuit", "investigation", "sec", "doj", "settlement"], "partnership": ["partnership", "collaboration", "acquisition", "merger"]}
    event_data = []
    for doc in all_docs:
        if any(re.search(r'\b' + kw + r'\b', doc.text, re.IGNORECASE) for keywords in EVENT_KEYWORDS.values() for kw in keywords):
            event_data.append({"date": doc.date, "text": doc.text})
    if not event_data: return pd.DataFrame()
    df = pd.DataFrame(event_data).drop_duplicates()
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=["event_sent_neg", "event_sent_neu", "event_sent_pos"])
    full_df = pd.concat([df.drop(columns="text"), sent_df], axis=1)
    daily_df = full_df.groupby("date").mean().reset_index()
    daily_df["is_event_day"] = 1
    return daily_df

def compute_attention_weights(row,
                              vol_col='vol_20',
                              vol_scale_minmax=None,
                              volume_col='volume',
                              event_col='is_event_day',
                              vol_pow=1.0,
                              volume_pow=1.0,
                              event_boost=2.0,
                              eps=1e-9):
    """
    Compute normalized attention weights for sources: news, reddit, yt, event.
    - row: pd.Series (single day)
    - vol_scale_minmax: optional tuple (min, max) to perform robust normalization (if None uses raw values)
    - vol_pow / volume_pow: exponents controlling sensitivity
    - event_boost: multiplier applied to sources when event day
    Returns dict: {'w_news':..., 'w_reddit':..., 'w_yt':..., 'w_event':...}
    """
    # Base signals for scaling
    vol = float(row.get(vol_col, 0.0) or 0.0)
    vol_score = max(vol, 0.0) ** vol_pow

    vol_norm = vol_score
    # volume scaling
    volu = float(row.get(volume_col, 0.0) or 0.0)
    volu_score = max(volu, 0.0) ** volume_pow

    # event indicator
    is_event = 1.0 if float(row.get(event_col, 0)) >= 1.0 else 0.0

    # basic heuristics: news weight driven by volatility, social (reddit/yt) by volume
    # create raw weights
    raw_news   = vol_norm + 0.1  # baseline
    raw_reddit = volu_score + 0.05
    raw_yt     = volu_score * 0.7 + 0.05
    raw_event  = (is_event * event_boost)  # if event, boost

    # optionally amplify news/social when event occurs
    if is_event:
        raw_news *= (1.0 + 0.5 * is_event)
        raw_event = raw_event * 1.0

    raw = np.array([raw_news, raw_reddit, raw_yt, raw_event], dtype=float)
    # avoid zero-sum
    if raw.sum() <= eps:
        raw = raw + 1.0

    weights = raw / raw.sum()
    return {
        "w_news": float(weights[0]),
        "w_reddit": float(weights[1]),
        "w_yt": float(weights[2]),
        "w_event": float(weights[3])
    }

def build_hybrid_featureset(stock_df, news_df, reddit_df, yt_df, event_df) -> pd.DataFrame:
    features = stock_df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)
    for text_df in [news_df, reddit_df, yt_df, event_df]:
        if not text_df.empty:
            text_df["date"] = pd.to_datetime(text_df["date"]).dt.tz_localize(None)
            features = pd.merge(features, text_df, on="date", how="left")
    features = add_technical_features(features)
    scaler = MinMaxScaler()
    features['vol_20_scaled'] = scaler.fit_transform(features[['vol_20']].fillna(0))
    features['volume_scaled'] = scaler.fit_transform(features[['volume']].fillna(0))
    features['is_event_day'] = features.get('is_event_day', 0).fillna(0)

    # --- Compute per-source sentiment ---
    features['news_sentiment_score'] = features.get('news_sent_pos', 0).fillna(0) - features.get('news_sent_neg', 0).fillna(0)
    features['reddit_sentiment_score'] = features.get('reddit_sent_pos', 0).fillna(0) - features.get('reddit_sent_neg', 0).fillna(0)
    features['yt_sentiment_score'] = features.get('yt_sent_pos', 0).fillna(0) - features.get('yt_sent_neg', 0).fillna(0)
    features['event_sentiment_score'] = features.get('event_sent_pos', 0).fillna(0) - features.get('event_sent_neg', 0).fillna(0)

    # --- Apply attention weighting ---
    weights_df = features.apply(lambda r: pd.Series(compute_attention_weights(r)), axis=1)
    features = pd.concat([features, weights_df], axis=1)

    features['combined_sentiment_signal'] = (
        features['w_news']   * features['news_sentiment_score'] +
        features['w_reddit'] * features['reddit_sentiment_score'] +
        features['w_yt']     * features['yt_sentiment_score'] +
        features['w_event']  * features['event_sentiment_score']
    )

    lag_periods = [1, 2, 3, 5, 10, 21, 63]
    lag_cols = [c for c in features.columns if c not in ['date','open','high','low','close'] and '_scaled' not in c]
    for lag in lag_periods:
        shifted = features[lag_cols].shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        features = pd.concat([features, shifted], axis=1)
    features['target'] = features['close'].shift(-1)
    return features.dropna(subset=["target", "close"]).fillna(0).reset_index(drop=True)

def build_residual_sequences(y_true, y_pred, lookback=10):
    residuals = y_true - y_pred
    X, y = [], []
    for i in range(len(residuals) - lookback):
        X.append(residuals.iloc[i:i+lookback].values)
        y.append(residuals.iloc[i+lookback])
    return np.array(X), np.array(y)

def train_residual_lstm(y_true, y_pred, lookback=10):
    X, y = build_residual_sequences(y_true, y_pred, lookback)
    if len(X) < 2: return None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([ LSTM(64, activation="tanh", input_shape=(lookback, 1), return_sequences=True), Dropout(0.2), LSTM(32, activation="tanh"), Dense(1) ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

def generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, query, predicted_price=None):
    if not GEMINI_API_KEY: return "Gemini API key not found."
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    texts = [f"[{d.source}] {d.text}" for d in (news_docs + reddit_docs + yt_docs)[:45]]
    extra = f"\nThe model predicts the next closing price will be around ${predicted_price:.2f}." if predicted_price else ""
    full_text = " ".join(texts)
    truncated_text = full_text[:10000] + ('...' if len(full_text) > 10000 else '')
    prompt = f"Summarize these financial signals for {ticker} ({','.join(query)}), focusing on key events, sentiment, and conflicts.\n\nRecent Texts:\n{truncated_text}\n{extra}"
    try:
        return model.generate_content(prompt).text
    except Exception as e: return f"Summary generation failed: {e}"

# ==============================================================================
#  SECTION 2: FLASK APPLICATION LOGIC
# ==============================================================================

def get_ticker_paths(ticker: str):
    """Generates all file paths for a given ticker."""
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
    """Computes a sophisticated recommendation based on multiple factors."""
    # 1. Price Prediction Factor (The most important signal)
    last_close = last_feature_row.get('close', 0)
    if last_close == 0: return "HOLD"
    price_pct_change = (last_pred - last_close) / last_close

    # 2. Technical Momentum Factor (Is the stock overbought/oversold?)
    rsi = last_feature_row.get('rsi_20', 50)  # Use 20-day RSI, default to neutral 50
    
    # 3. Trend Factor (Is the stock in an uptrend or downtrend?)
    sma_20 = last_feature_row.get('sma_20', last_close)
    sma_60 = last_feature_row.get('sma_60', last_close)
    in_uptrend = sma_20 > sma_60

    # 4. Sentiment Factor (What is the news/social media saying?)
    news_att = last_feature_row.get('news_attention_score', 0)
    reddit_att = last_feature_row.get('reddit_attention_score', 0)
    event_sent = last_feature_row.get('event_sent_pos', 0) - last_feature_row.get('event_sent_neg', 0)
    total_sentiment = news_att + reddit_att + event_sent

    # --- New Rule-Based Logic ---
    if price_pct_change > 0.02:
        return "STRONG BUY"
    if price_pct_change < -0.02:
        return "STRONG SELL"

    # For smaller predicted changes, use confirming factors
    if price_pct_change > 0.005: # Small predicted increase
        # Confirm with trend and sentiment
        if in_uptrend and total_sentiment > 0.05:
            return "BUY"
        # Contradicted by strong negative sentiment or being overbought
        if total_sentiment < -0.1 or rsi > 75:
            return "HOLD"
        return "BUY" # Default to buy if prediction is positive

    if price_pct_change < -0.005: # Small predicted decrease
        # Confirm with downtrend and sentiment
        if not in_uptrend and total_sentiment < -0.05:
            return "SELL"
        # Contradicted by strong positive sentiment or being oversold
        if total_sentiment > 0.1 or rsi < 25:
            return "HOLD"
        return "SELL" # Default to sell if prediction is negative

    return "HOLD" # Default case for very small/flat predictions


def align_texts_to_market_cutoff(docs: List[Doc], market_close_hour=16):
    """Keep docs only if published before market close of that day."""
    out = []
    for d in docs:
        if isinstance(d.date, pd.Timestamp):
            cutoff = d.date.normalize() + pd.Timedelta(hours=market_close_hour)
            if d.date <= cutoff:
                out.append(d)
        else:
            out.append(d)
    return out

def run_full_training(ticker: str, queries: list, days: int):
    """The main, unified training function that runs your entire pipeline."""
    paths = get_ticker_paths(ticker)
    
    init_nlp()
    stock_df = get_stock_data(ticker, days=days)
    news_docs = fetch_news_rss(queries, days=days)
    reddit_docs = fetch_reddit_praw(queries, days=days)
    yt_docs = fetch_youtube_docs(queries, days=days)

    # Prevent lookahead leakage
    news_docs = align_texts_to_market_cutoff(news_docs)
    reddit_docs = align_texts_to_market_cutoff(reddit_docs)
    yt_docs = align_texts_to_market_cutoff(yt_docs)

    all_docs = news_docs + reddit_docs + yt_docs

    news_daily_features = aggregate_text_features(news_docs, "news")
    reddit_daily_features = aggregate_text_features(reddit_docs, "reddit")
    yt_daily_features = aggregate_text_features(yt_docs, "yt")
    event_daily_features = detect_event_features(all_docs)

    final_df = build_hybrid_featureset(
        stock_df, news_daily_features, reddit_daily_features,
        yt_daily_features, event_daily_features
    )
    if final_df.empty:
        raise RuntimeError("Dataframe is empty after feature engineering.")

    split_index = int(len(final_df) * 0.8)
    train_df, test_df = final_df.iloc[:split_index], final_df.iloc[split_index:]
    
    X_train = train_df.drop(columns=['date', 'target', 'close'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['date', 'target', 'close'])
    y_test = test_df['target']
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=RANDOM_SEED)
    model.fit(X_train_scaled_df, y_train, eval_set=[(X_test_scaled_df, y_test)], eval_metric='l1', callbacks=[early_stopping(100, verbose=False)])

    preds_train = model.predict(X_train_scaled_df)
    preds_test = model.predict(X_test_scaled_df)

    # --- Train LSTM on TRAIN residuals only ---
    residuals_train = y_train.reset_index(drop=True) - pd.Series(preds_train)
    lookback = 10
    residual_lstm = train_residual_lstm(residuals_train, pd.Series(preds_train), lookback=lookback)

    corrected_preds = preds_test.copy()
    if residual_lstm is not None:
        # Seed residuals with last training window
        seed = residuals_train.values[-lookback:] if len(residuals_train) >= lookback else np.zeros(lookback)
        X_seed = seed.reshape(1, lookback, 1)
        resid_preds = []
        for i in range(len(preds_test)):
            pred = residual_lstm.predict(X_seed, verbose=0).flatten()[0]
            resid_preds.append(pred)
            X_seed = np.roll(X_seed, -1, axis=1)
            X_seed[0, -1, 0] = pred
        corrected_preds = preds_test + np.array(resid_preds)
        residual_lstm.save(paths["lstm"])

    
    final_mae = mean_absolute_error(y_test, corrected_preds)
    final_r2 = r2_score(y_test, corrected_preds)

    joblib.dump(model, paths["lgbm"])
    joblib.dump(scaler, paths["scaler"])
    meta = {"feature_names": list(X_train.columns), "lookback": 10, "mae_train": final_mae, "r2_train": final_r2, "training_date": datetime.utcnow().isoformat()}
    joblib.dump(meta, paths["meta"])

    last_pred = float(corrected_preds[-1])
    summary = generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, queries, last_pred)
    with open(paths["summary"], "w", encoding='utf-8') as f: f.write(summary)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(test_df['date'], y_test, label="Actual Price", color="royalblue")
    ax.plot(test_df['date'], preds_test, label="LGBM Prediction", color="darkorange", ls="--")
    if residual_lstm:
        ax.plot(test_df['date'], corrected_preds, label="Hybrid (LGBM + LSTM)", color="green", ls="-.")
    ax.set_title(f"{ticker} Price Prediction", fontsize=16)
    ax.legend()
    fig.tight_layout()
    plt.savefig(paths["plot"])
    plt.close(fig)

    signal = compute_advanced_signal(test_df.iloc[-1], last_pred)
    return {"last_pred": last_pred, "signal": signal}

def predict_fast(ticker: str):
    """Loads artifacts and predicts using only the latest stock data."""
    paths = get_ticker_paths(ticker)
    
    model = joblib.load(paths["lgbm"])
    scaler = joblib.load(paths["scaler"])
    meta = joblib.load(paths["meta"])
    feature_names = meta["feature_names"]
    
    days_to_fetch = 120 + 63 + 10 
    stock_df = get_stock_data(ticker, days_to_fetch)
    
    features_df = build_hybrid_featureset(stock_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    last_row = features_df.tail(1)
    
    X_predict = pd.DataFrame(columns=feature_names)
    X_predict_data = last_row.drop(columns=['date', 'target', 'close'], errors='ignore')
    X_predict = pd.concat([X_predict, X_predict_data], ignore_index=True).fillna(0)
    X_predict = X_predict[feature_names]

    X_scaled = scaler.transform(X_predict)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    prediction = model.predict(X_scaled_df)[0]
    
    signal = compute_advanced_signal(last_row.iloc[0], prediction)
    return {"last_pred": prediction, "signal": signal}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not _lock.acquire(blocking=False):
        flash("Another process is already running. Please wait.", "error")
        return redirect(url_for("index"))
    
    try:
        ticker = request.form.get("ticker", "").strip().upper()
        queries = [q.strip() for q in request.form.get("queries", "").split(",")] if request.form.get("queries") else [ticker]
        days = int(request.form.get("days", 1200))
        retrain = "retrain" in request.form

        if not ticker:
            flash("Ticker symbol is required.", "error")
            return redirect(url_for("index"))

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

        return render_template("result.html",
                               ticker=ticker,
                               results=results,
                               meta=meta,
                               summary=summary,
                               plot_url=paths["plot_url"])

    except Exception as e:
        traceback.print_exc()
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for("index"))
    finally:
        _lock.release()

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    with open("templates/layout.html", "w") as f:
        f.write("""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multimodal Stock Predictor</title><link rel="stylesheet" href="/static/style.css"></head><body><div class="container">
<header><h1>📈 Multimodal Stock Predictor</h1></header><main>
{% with messages = get_flashed_messages(with_categories=true) %}{% if messages %}{% for category, message in messages %}
<div class="flash {{ category }}">{{ message }}</div>{% endfor %}{% endif %}{% endwith %}
{% block content %}{% endblock %}</main><footer>
<p>Powered by LightGBM, LSTM, and Multimodal Data Fusion</p></footer></div></body></html>
        """)

    with open("templates/index.html", "w") as f:
        f.write("""
{% extends "layout.html" %}{% block content %}<div class="card">
<form method="post" action="/predict"><div class="form-group"><label for="ticker">Ticker Symbol</label>
<input id="ticker" name="ticker" value="AAPL" required placeholder="e.g., AAPL, GOOG, NAZARA.NS"></div>
<div class="form-group"><label for="queries">Search Queries (comma-separated)</label>
<input id="queries" name="queries" value="Apple, Iphone, Tim cook" placeholder="Used for News, Reddit, YouTube"></div>
<div class="form-group"><label for="days">Training History (Days)</label><input id="days" name="days" type="number" value="1200"></div>
<div class="form-group checkbox-group"><input type="checkbox" id="retrain" name="retrain">
<label for="retrain">Force Full Retraining (slower, full analysis)</label></div>
<button type="submit">Analyze & Predict</button></form></div>{% endblock %}
        """)

    with open("templates/result.html", "w") as f:
        f.write("""
{% extends "layout.html" %}{% block content %}<div class="card result-card">
<h2>Results for {{ ticker }}</h2><div class="signal-box">Prediction: <strong>${{ "%.2f"|format(results.last_pred) }}</strong>
<span class="signal {{ results.signal|lower|replace(' ', '-') }}">{{ results.signal }}</span></div>
{% if meta.mae_train %}<div class="metrics"><h4>Last Full Training Performance</h4>
<p><strong>MAE:</strong> {{ "%.4f"|format(meta.mae_train) }}</p><p><strong>R² Score:</strong> {{ "%.4f"|format(meta.r2_train) }}</p>
<p class="muted">Trained on: {{ meta.training_date.split('T')[0] }}</p></div>{% endif %}
{% if plot_url and meta.mae_train %}<div class="plot"><h3>Prediction Chart</h3><img src="{{ plot_url }}?t={{ range(1, 100000) | random }}" alt="Prediction plot for {{ ticker }}">
</div>{% endif %}<div class="summary"><h3>AI-Generated Summary</h3><pre>{{ summary or 'Summary not available for fast predictions.' }}</pre></div>
<a href="/" class="back-link">← Make another prediction</a></div>{% endblock %}
        """)

    with open("static/style.css", "w") as f:
        f.write("""
:root { --bg: #EDEDED; --card-bg: #fff; --text: #334155; --heading: #1e293b; --border: #e2e8f0; --shadow: 0 10px 25px -5px rgba(0,0,0,0.07); --primary: #4f46e5; --primary-light: #e0e7ff; --buy: #16a34a; --sell: #dc2626; --hold: #64748b; --info-bg: #eff6ff; --info-text: #1d4ed8; --error-bg: #fef2f2; --error-text: #991b1b; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }
.container { max-width: 900px; margin: 0 auto; }
header h1 { color: var(--heading); text-align: center; font-size: 2rem; margin-bottom: 2rem; }
.card { background: var(--card-bg); border-radius: 12px; padding: 2rem; box-shadow: var(--shadow); }
.form-group { margin-bottom: 1.5rem; }
label { font-weight: 600; display: block; margin-bottom: 0.5rem; }
input[type="text"], input[type="number"] { width: 100%; padding: 0.75rem; border: 1px solid var(--border); border-radius: 8px; font-size: 1rem; box-sizing: border-box; transition: all 0.2s ease; }
input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-light); }
.checkbox-group { display: flex; align-items: center; }
.checkbox-group input { margin-right: 0.5rem; width: auto; height: 1em; width: 1em; }
button { width: 100%; padding: 1rem; background: var(--primary); color: #fff; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: background 0.2s ease; }
button:hover { background: #4338ca; }
.flash { padding: 1rem; margin-bottom: 1rem; border-radius: 8px; border: 1px solid transparent; }
.flash.info { background: var(--info-bg); color: var(--info-text); border-color: #bfdbfe; }
.flash.error { background: var(--error-bg); color: var(--error-text); border-color: #fecaca; }
.result-card h2 { text-align: center; color: var(--heading); }
.signal-box { text-align: center; font-size: 1.5rem; margin-bottom: 2rem; background: var(--bg); padding: 1rem; border-radius: 8px; }
.signal { font-weight: bold; color: #fff; padding: 0.5rem 1.5rem; border-radius: 50px; margin-left: 1rem; text-transform: uppercase; font-size: 1.2rem;}
.signal.buy { background: var(--buy); }
.signal.strong-buy { background: #14532d; }
.signal.sell { background: var(--sell); }
.signal.strong-sell { background: #7f1d1d; }
.signal.hold { background: var(--hold); }
.metrics { background: var(--bg); padding: 1rem; border-radius: 8px; text-align: center; margin: 2rem 0; }
.metrics p { margin: 0.5rem 0; }
.muted { color: #64748b; font-size: 0.9rem; }
.plot img { max-width: 100%; border-radius: 8px; margin-top: 1.5rem; border: 1px solid var(--border); }
.summary { margin-top: 2rem; }
.summary h3, .plot h3 { color: var(--heading); }
.summary pre { background: #1e293b; color: #e2e8f0; padding: 1.5rem; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem; line-height: 1.6; }
.back-link { display: inline-block; margin-top: 2rem; font-weight: 600; color: var(--primary); }
footer { text-align: center; margin-top: 2rem; color: #94a3b8; }
        """)

    app.run(host="0.0.0.0", port=5691, debug=True)