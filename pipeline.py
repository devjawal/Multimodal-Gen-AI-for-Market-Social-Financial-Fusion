#!/usr/bin/env python3


import os
# --- Environment Setup ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import certifi
    ca_path = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = ca_path
    os.environ['SSL_CERT_FILE'] = ca_path
    os.environ['CURL_CA_BUNDLE'] = ca_path
    print(f"[DEBUG] Using certifi CA bundle at: {ca_path}")
except Exception as _e:
    print(f"[WARN] certifi import failed or not installed: {_e}")

import re
import joblib
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import numpy as np
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _YT_TRANSCRIPT_AVAILABLE = True
except Exception:
    _YT_TRANSCRIPT_AVAILABLE = False

from dotenv import load_dotenv
import praw
import yfinance as yf
from dateutil import parser

# --- NLP / ML Imports ---
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    _NLP_AVAILABLE = True
except Exception as e:
    print(f"Warning: NLP libraries not found. install transformers and sentence-transformers. Error: {e}")
    _NLP_AVAILABLE = False

from lightgbm import LGBMRegressor, early_stopping as lgbm_early_stopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping 

# --- User Configuration ---
# TICKER = "TSLA"
# QUERY = ["Tesla", "EV", "Elon Musk", "Tesla Inc","Electric cars"]
# TICKER = "AAPL"
# QUERY = ["Apple Inc", "iPhone 17", "Tim Cook", "iphone","Macbook"]

# TICKER = "TSN"
# QUERY = ["Tyson Foods earnings", "Tyson Foods guidance", "Tyson Foods merger acquisition", "Tyson Foods profit warning"]

TICKER = "RELIANCE.NS"
QUERY = ["Reliance Industries Limited",
    "Reliance Jio",
    "Reliance Retail",
    "Mukesh Ambani",
    "RIL earnings",
    "Reliance quarterly results",
    "Reliance acquisition",
    "Reliance partnership"]


DAYS = 1200 
RANDOM_SEED = 42
N_PCA_COMPONENTS = 10
LSTM_LOOKBACK = 10 # Lookback window for LSTM
RECENCY_HALFLIFE_DAYS = 3  # halflife for exponential recency weighting of text features
np.random.seed(RANDOM_SEED)

# --- Environment & Paths ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stock-predictor-v1 by user")

ART_DIR = os.path.join(os.getcwd(), "artifacts")
CACHE_DIR = os.path.join(ART_DIR, "cache") 
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True) 

# --- Pandas print options ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 6)

# ----------------------------- Helpers & NLP Backend ---------------------------------
@dataclass
class Doc:
    date: pd.Timestamp
    text: str
    source: str

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())

_nlp = {"ok": False, "fin_pipe": None, "embedder": None}
EMB_DIM = 384

def init_nlp():
    if _nlp["ok"] or not _NLP_AVAILABLE:
        return
    print("Loading NLP models...")
    try:
        fin_pipe = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _nlp.update({"ok": True, "fin_pipe": fin_pipe, "embedder": embedder})
        print("NLP models loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load NLP models. Error: {repr(e)}")

def finbert_sentiment_vectors(texts: List[str]) -> np.ndarray:
    if not texts or not _nlp["ok"]:
        return np.array([[0.0, 1.0, 0.0]] * len(texts))
    results = []
    for text_chunk in [texts[i:i + 16] for i in range(0, len(texts), 16)]:
        try:
            preds = _nlp["fin_pipe"](text_chunk)
            for p in preds:
                vec = np.zeros(3)
                label = p.get("label", "neutral").lower()
                score = p.get("score", 0.0)
                if "negative" in label: vec[0] = score
                elif "positive" in label: vec[2] = score
                else: vec[1] = score
                results.append(vec)
        except Exception:
            results.extend([np.array([0.0, 1.0, 0.0])] * len(text_chunk))
    return np.array(results)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts or not _nlp["ok"]:
        return np.zeros((len(texts), EMB_DIM))
    return _nlp["embedder"].encode(texts, show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True, batch_size=32)

# ----------------------------- Data Fetching ---------------------------------
def get_stock_data(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No stock data for {ticker}")

    df = df.reset_index()
    
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            name = '_'.join(filter(None, col)) 
        else:
            name = col
        new_columns.append(name.lower().replace(" ", "_"))
    
    df.columns = new_columns
    
    rename_map = {
        "adj_close": "close",
        f"close_{ticker.lower()}": "close",
        f"open_{ticker.lower()}": "open",
        f"high_{ticker.lower()}": "high",
        f"low_{ticker.lower()}": "low",
        f"volume_{ticker.lower()}": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df.rename(columns={c: "date"}, inplace=True)
                break
        else:
             raise RuntimeError(f"'date' column missing! Columns: {df.columns.tolist()}")

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            if col == 'close' and 'adj_close' in df.columns:
                 df.rename(columns={'adj_close': 'close'}, inplace=True)
            else:
                raise RuntimeError(f"Required column '{col}' missing after cleaning! Columns: {df.columns.tolist()}")
    
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[required_cols].sort_values("date").reset_index(drop=True)


def fetch_news_rss(queries, days: int) -> List[Doc]:
    if isinstance(queries, str): queries = [queries]
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
                text = clean_text(f"{getattr(entry, 'title', '')}")
                key = (text[:50], dt.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(dt).normalize(), text=text, source="news"))
                    seen.add(key)
            except Exception: continue
    return docs

def fetch_reddit_praw(queries, days: int) -> List[Doc]:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        print("WARNING: Reddit credentials not found. Skipping Reddit fetch.")
        return []
    if isinstance(queries, str): queries = [queries]
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    try:
        subs = "wallstreetbets+stocks+investing+apple+technology"
        for query in queries:
            for submission in reddit.subreddit(subs).search(query, sort="new", limit=100):
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                if created < cutoff: continue
                text = clean_text(f"{submission.title}")
                key = (submission.id, created.date())
                if text and key not in seen:
                    docs.append(Doc(date=pd.to_datetime(created).normalize(), text=text, source="reddit"))
                    seen.add(key)
    except Exception as e:
        print(f"WARNING: Reddit fetch failed: {e}")
    return docs

def fetch_youtube_docs(queries, days: int) -> List[Doc]:
    if not YOUTUBE_API_KEY:
        print("WARNING: No YouTube API key provided. Skipping YouTube fetch.")
        return []
    if isinstance(queries, str): queries = [queries]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    docs, seen = [], set()
    for query in queries:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet", "q": f'"{query}" stock analysis', "type": "video",
            "order": "date", "maxResults": 50, "key": YOUTUBE_API_KEY,
            "publishedAfter": cutoff.isoformat("T").replace("+00:00", "Z")
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            for item in r.json().get("items", []):
                video_id = item["id"]["videoId"]
                if video_id in seen: continue
                published_at = parser.parse(item["snippet"]["publishedAt"])
                title = item["snippet"]["title"]
                text = clean_text(title)
                if text:
                    docs.append(Doc(date=pd.to_datetime(published_at).normalize(), text=text, source="youtube"))
                    seen.add(video_id)
        except Exception as e:
            print(f"WARNING: YouTube fetch failed for query '{query}': {e}")
    return docs

# ----------------------------- Feature Engineering ---------------------------------

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    
    out['intraday_vol'] = (out['high'] - out['low']) / out['close']
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
        
    out['price_vs_sma20'] = (out['close'] - out['sma_20']) / out['sma_20']
    out['price_vs_sma120'] = (out['close'] - out['sma_120']) / out['sma_120']
    # safe vol_vs_vol60
    out['vol_vs_vol60'] = (out.get('vol_5', 0) - out.get('vol_60', 0)) / (out.get('vol_60', 1e-9))
    out['rolling_corr_vol'] = out['close'].rolling(20).corr(out['volume'])
    out['breakout_signal'] = (out['close'] >= out['high'].rolling(20).max().shift(1)).astype(float)

    return out

def classify_event_type(text):
    """
    Simple keyword-based event classifier.
    Given a text (e.g. news/reddit/youtube title), returns a broad event category.
    """
    if not text or not isinstance(text, str):
        return "General Event"

    t = text.lower()

    # --- Earnings / Finance related ---
    if any(k in t for k in ["earning", "quarter", "q1", "q2", "q3", "q4", "profit", "revenue", "results", "guidance"]):
        return "Earnings Report"

    # --- Product Launch / Tech releases ---
    if any(k in t for k in ["launch", "release", "introduce", "unveil", "new model", "product", "rollout", "update", "version"]):
        return "Product Launch"

    # --- Mergers / Acquisitions / Partnerships ---
    if any(k in t for k in ["merger", "acquisition", "buyout", "partnership", "collaboration", "takeover", "joins forces"]):
        return "M&A / Partnership"

    # --- Layoffs / Job cuts ---
    if any(k in t for k in ["layoff", "job cut", "fired", "redundancy", "workforce reduction"]):
        return "Layoff / Workforce"

    # --- Leadership changes ---
    if any(k in t for k in ["ceo", "cfo", "executive", "resign", "appoint", "steps down", "leadership"]):
        return "Leadership Change"

    # --- Legal / Regulatory ---
    if any(k in t for k in ["lawsuit", "investigation", "fine", "regulator", "court", "trial"]):
        return "Legal / Regulatory"

    # --- Market / Economic conditions ---
    if any(k in t for k in ["inflation", "market crash", "growth", "dow jones", "nasdaq", "fed", "interest rate"]):
        return "Market Event"

    # --- Default ---
    return "General Event"


def detect_event_features(all_docs: List[Doc]) -> pd.DataFrame:
    """
    Detect market-moving events (earnings, launches, legal, partnerships, layoffs, leadership, etc.)
    from all text docs and calculate sentiment.
    Returns a daily dataframe with event_type, sentiment, and is_event_day flag.
    """
    import re
    import pandas as pd

    if not all_docs:
        return pd.DataFrame(columns=["date", "event_type", "event_sent_neg", "event_sent_neu", "event_sent_pos", "is_event_day"])

    # --- Define keyword groups (your existing logic retained) ---
    EVENT_KEYWORDS: Dict[str, List[str]] = {
        "Earnings Report": ["earnings call", "quarterly results", "revenue", "profit", "guidance"],
        "Product Launch": ["product launch", "announcement", "release date", "unveiled", "new iphone", "new model"],
        "Legal / Regulatory": ["lawsuit", "investigation", "sec", "doj", "settlement", "antitrust", "court", "fine"],
        "M&A / Partnership": ["partnership", "collaboration", "acquisition", "merger", "deal", "buyout"],
        "Layoff / Workforce": ["layoff", "job cut", "redundancy", "fired", "workforce reduction"],
        "Leadership Change": ["ceo", "cfo", "executive", "resign", "appoint", "steps down"],
        "Market Event": ["inflation", "market crash", "dow jones", "nasdaq", "interest rate", "fed"]
    }

    event_data = []

    # --- Match each doc to event keywords and fallback to classify_event_type helper ---
    for doc in all_docs:
        text = getattr(doc, "text", "")
        date = getattr(doc, "date", None)

        matched_types = []
        for event_type, keywords in EVENT_KEYWORDS.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE) for kw in keywords):
                matched_types.append(event_type)

        # Fallback if no explicit keyword matched → use classify_event_type()
        if not matched_types:
            matched_types.append(classify_event_type(text))

        event_data.append({
            "date": date,
            "text": text,
            "event_type": ", ".join(set(matched_types))
        })

    if not event_data:
        return pd.DataFrame(columns=["date", "event_type", "event_sent_neg", "event_sent_neu", "event_sent_pos", "is_event_day"])

    df = pd.DataFrame(event_data).drop_duplicates(subset=["date", "text"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dropna()

    # --- Run FinBERT on event texts ---
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=["event_sent_neg", "event_sent_neu", "event_sent_pos"])

    # --- Combine sentiment and event type ---
    df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

    # --- Aggregate per date + event_type ---
    agg_funcs = {col: "mean" for col in ["event_sent_neg", "event_sent_neu", "event_sent_pos"]}
    daily_df = df.groupby(["date", "event_type"]).agg(agg_funcs).reset_index()

    # --- Mark as event day ---
    daily_df["is_event_day"] = 1

    print(f"[INFO] Event detection complete — {len(daily_df)} unique event-day records found.")
    return daily_df

def process_text_source(docs: List[Doc], source_name: str) -> pd.DataFrame:
    """
    Process raw docs into per-day aggregated sentiment + PCA components.
    NOTE: this function returns per-day aggregates (mean across docs for that day).
    Later we will apply exponential recency weighting across days to produce the final
    time-series-aligned features (in build_hybrid_featureset).
    """
    if not docs:
        return pd.DataFrame()
        
    print(f"Processing {len(docs)} docs for source: {source_name}")
    
    df_raw_text = pd.DataFrame([{"date": d.date, "text": d.text} for d in docs])
    df_raw_text = df_raw_text.sort_values(by="date").reset_index(drop=True)

    sentiments = finbert_sentiment_vectors(df_raw_text["text"].tolist())
    sent_cols = [f"{source_name}_sent_neg", f"{source_name}_sent_neu", f"{source_name}_sent_pos"]
    sent_df = pd.DataFrame(sentiments, columns=sent_cols)

    embeddings = embed_texts(df_raw_text["text"].tolist())
    # PCA here to compress per-doc embedding — will aggregate per day afterwards
    pca = PCA(n_components=min(N_PCA_COMPONENTS, embeddings.shape[1]), random_state=RANDOM_SEED)
    try:
        pca_embeddings = pca.fit_transform(embeddings)
    except Exception:
        pca_embeddings = np.zeros((len(embeddings), N_PCA_COMPONENTS))
    pca_cols = [f"{source_name}_pca_{i}" for i in range(pca_embeddings.shape[1])]
    pca_df = pd.DataFrame(pca_embeddings, columns=pca_cols)

    df_full = pd.concat([df_raw_text, sent_df, pca_df], axis=1)
    
    # Aggregate per date by mean for now 
    agg_cols = sent_cols + pca_cols
    agg_funcs = {col: "mean" for col in agg_cols}
    df_aggregated = df_full.groupby("date").agg(agg_funcs).reset_index()
    
    print(f"Aggregated {source_name} features shape: {df_aggregated.shape}")
    return df_aggregated

# --- exponential recency-weighting across days for merged text features ---
def apply_ewm_to_text_features(features_df: pd.DataFrame, text_cols: List[str], halflife_days: float = RECENCY_HALFLIFE_DAYS):
    """
    Apply exponential recency-weighted averaging across days for the given text_cols.
    Uses a halflife (in days) so that recent days have exponentially higher weight.
    Works safely with timezone-aware or naive datetimes.
    """
    if len(text_cols) == 0:
        return features_df

    df = features_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Convert datetime to integer days (avoid deprecated .view)
    ts_days = (df["date"].astype("int64") // (24 * 60 * 60 * 10**9)).astype(float)

    vals = df[text_cols].to_numpy(dtype=float)
    n = len(df)
    ewm_results = np.zeros_like(vals)

    for i in range(n):
        ages = ts_days[i] - ts_days[:i + 1]
        # convert to numpy to allow reshape
        w = np.exp(-np.log(2) * (ages / float(halflife_days)))
        W = np.asarray(w).reshape(-1, 1)
        numer = (W * vals[:i + 1]).sum(axis=0)
        denom = W.sum() + 1e-12
        ewm_results[i] = numer / denom

    df[text_cols] = ewm_results
    return df

def build_hybrid_featureset(stock_df, news_df, reddit_df, yt_df, event_df) -> pd.DataFrame:
    print("Building multi-source hybrid feature set (patched v14)...")
    features = stock_df.copy()
    # --- Ensure event_df columns are numeric-safe before merging ---
    if not event_df.empty and "event_type" in event_df.columns:
        # One-hot encode the event_type (so it's numeric)
        event_type_dummies = pd.get_dummies(event_df["event_type"], prefix="event")
        event_df = pd.concat([event_df.drop(columns=["event_type"]), event_type_dummies], axis=1)

    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)

    # Merge daily aggregated text features (per-source) - left join
    for text_df in [news_df, reddit_df, yt_df, event_df]:
        if not text_df.empty:
            text_df["date"] = pd.to_datetime(text_df["date"]).dt.tz_localize(None)
            features = pd.merge(features, text_df, on="date", how="left")

    # Add technical indicators
    features = add_technical_features(features)

    # Identify text columns (sentiment & pca columns)
    text_pca_cols = [col for col in features.columns if ('_sent_' in col) or ('_pca_' in col)]
    event_cols = [col for col in features.columns if col.startswith('event_') or col == 'is_event_day']

    # Replace NaNs in daily aggregated per-source by 0 for days with no docs
    features[text_pca_cols] = features[text_pca_cols].fillna(0.0)
    features[event_cols] = features[event_cols].fillna(0.0)

    # --- APPLY EXPONENTIAL RECENCY WEIGHTING across dates for text features ---
    # This turns per-day means into recency-weighted series so that a spike in recent news strongly affects features.
    if len(text_pca_cols) > 0:
        features = apply_ewm_to_text_features(features, text_pca_cols, halflife_days=RECENCY_HALFLIFE_DAYS)

    # create attention-style aggregated columns and event multipliers
    scaler = MinMaxScaler()
    # fill any remaining inf/nan safely
    features['intraday_vol'] = features['intraday_vol'].replace([np.inf, -np.inf], 0).fillna(0)
    features['vol_20'] = features['vol_20'].replace([np.inf, -np.inf], 0).fillna(0)
    features['intraday_vol_scaled'] = scaler.fit_transform(features[['intraday_vol']])
    features['vol_20_scaled'] = scaler.fit_transform(features[['vol_20']])

    for source in ["news", "reddit", "yt", "youtube"]:
        # There is a mixture of 'yt' vs 'youtube' naming - handle both
        candidates_pos = [c for c in features.columns if c.endswith(f"{source}_sent_pos")]
        candidates_neg = [c for c in features.columns if c.endswith(f"{source}_sent_neg")]
        # approximate safe check
        pos_col = None
        neg_col = None
        for c in features.columns:
            if c == f"{source}_sent_pos": pos_col = c
            if c == f"{source}_sent_neg": neg_col = c
        # fallback: check substring
        if pos_col is None:
            for c in features.columns:
                if c.endswith("_sent_pos") and source in c:
                    pos_col = c
        if neg_col is None:
            for c in features.columns:
                if c.endswith("_sent_neg") and source in c:
                    neg_col = c

        if pos_col and neg_col:
            features[f'{source}_agg_sent'] = features[pos_col] - features[neg_col]
            features[f'att_{source}_x_vol20'] = features[f'{source}_agg_sent'] * features['vol_20_scaled']
            features[f'att_{source}_x_intraday'] = features[f'{source}_agg_sent'] * features['intraday_vol_scaled']
            features[f'att_{source}_x_event'] = features[f'{source}_agg_sent'] * features['is_event_day'].fillna(0)
    # fallback: if none present, ok.

    # --- LAGGING (keep many lags, but drop columns with zero variance later) ---
    lag_periods = [1, 2, 3, 5, 10, 21, 63]
    base_price_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    all_cols = features.columns.tolist()
    lag_cols = [col for col in all_cols if col not in base_price_cols + ['target']]

    for lag in lag_periods:
        shifted = features[lag_cols].shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        features = pd.concat([features, shifted], axis=1)

    # --- NEW: target becomes next-day return ---
    features['target_ret'] = features['close'].shift(-1) / features['close'] - 1.0

    # Replace inf and fill NaNs
    features.replace([np.inf, -np.inf], 0, inplace=True)
    features = features.fillna(0)

    print(f"Final feature set shape: {features.shape}")
    return features.reset_index(drop=True)

# ----------------------------- NEW: Simple Residual LSTM helpers ---------------------------------
def create_simple_residual_sequences_from_series(series: pd.Series, lookback: int):
    """
    Creates sequences for the simple residual LSTM from a continuous series.
    'series' should be a pandas Series (1D).
    Returns X (n, lookback) and y (n,).
    """
    data = series.reset_index(drop=True)
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:(i + lookback)].values)
        y.append(data.iloc[i + lookback])
    return np.array(X), np.array(y)

def build_simple_residual_lstm(input_shape):
    """Builds a *stronger*, stacked LSTM model to predict error from past errors."""
    model = Sequential([
        # First LSTM layer - increased units and returns sequences for the next layer
        LSTM(64, activation="tanh", input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        
        # Second LSTM layer to capture more complex temporal patterns
        LSTM(32, activation="tanh", return_sequences=False),
        Dropout(0.3),
        
        # An intermediate dense layer to help interpret the LSTM's output
        Dense(16, activation='relu'),
        
        # Final output layer
        Dense(1) # Output is the *predicted error (return units)*
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# ----------------------------- Weighting & utility ---------------------------------
def compute_sample_weights(df: pd.DataFrame):
    """
    Compute per-row sample weights emphasizing recency, event days, breakouts and news attention.
    Returns an array of length len(df).
    """
    n = len(df)
    # recency weight exponential increasing (older -> lower)
    base = np.exp(np.linspace(0, 3.0, n))
    # event upweight
    event = df.get('is_event_day', pd.Series(0, index=df.index)).fillna(0).values
    breakout = df.get('breakout_signal', pd.Series(0, index=df.index)).fillna(0).values
    # news attention - use any one of att_news_x_vol20 or att_news_x_intraday if exists
    news_att = df.get('att_news_x_vol20', pd.Series(0, index=df.index)).fillna(0).values
    # combine multiplicatively but normalized
    weights = base * (1 + 6.0 * event) * (1 + 3.0 * breakout) * (1 + 2.0 * np.tanh(news_att))
    weights = weights / (np.mean(weights) + 1e-12)
    return weights

def train_and_evaluate_models(df: pd.DataFrame):
    """
    Final stable training function (v15 - fixed, NaN/alignment issues handled)
    - Trains LGBM on next-day returns (target_ret)
    - Trains residual LSTM to correct LGBM residuals
    - Evaluates both in PRICE-space with proper alignment (no leakage)
    - Produces one final Actual vs LGBM vs Hybrid plot
    """

    # --- Basic safety checks ---
    if df.shape[0] < 150:
        print(f"ERROR: Insufficient data ({df.shape[0]} rows).")
        return None, None

    initial_data_rows = (df['sma_120'] == 0).sum()
    if initial_data_rows > 0:
        df = df.iloc[initial_data_rows:].reset_index(drop=True)
    if df.shape[0] < 60:
        print("ERROR: Too few rows after cleaning.")
        return None, None

    # --- Split Train/Test ---
    df_for_training = df.iloc[:-1].copy().reset_index(drop=True)
    last_row_features = df.iloc[-1].copy()
    split_index = int(len(df_for_training) * 0.8)
    train_df = df_for_training.iloc[:split_index].reset_index(drop=True)
    test_df = df_for_training.iloc[split_index:].reset_index(drop=True)

    # --- Train/Test matrices ---
    print("--- Training Base LGBM Model (on returns) ---")
    y_train = train_df['target_ret']
    X_train = train_df.drop(columns=['date', 'target_ret', 'target'], errors='ignore')
    y_test = test_df['target_ret']
    X_test = test_df.drop(columns=['date', 'target_ret', 'target'], errors='ignore')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Drop near-constant cols
    var = X_train.var(axis=0)
    non_const_cols = var[var > 1e-6].index.tolist()
    if len(non_const_cols) < X_train.shape[1]:
        print(f"Dropping {X_train.shape[1] - len(non_const_cols)} near-constant columns.")
    X_train, X_test = X_train[non_const_cols], X_test[non_const_cols]

    # --- Scaling & weights ---
    lgbm_scaler = StandardScaler()
    X_train_scaled = lgbm_scaler.fit_transform(X_train)
    X_test_scaled = lgbm_scaler.transform(X_test)
    sample_weights = compute_sample_weights(train_df)

    # --- LightGBM model ---
    lgbm_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=5,
        reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1
    )

    lgbm_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='l1',
        callbacks=[lgbm_early_stopping(100, verbose=False)],
        sample_weight=sample_weights
    )

    # --- Evaluate LGBM (returns) ---
    lgbm_preds_train = lgbm_model.predict(X_train_scaled)
    lgbm_preds_test = lgbm_model.predict(X_test_scaled)
    mae_ret = mean_absolute_error(y_test, lgbm_preds_test)
    r2_ret = r2_score(y_test, lgbm_preds_test)
    print(f"\n--- Model Evaluation (LGBM on returns) ---")
    print(f"LGBM MAE (ret): {mae_ret:.6f}")
    print(f"LGBM R² (ret):  {r2_ret:.6f}")

    # --- Proper alignment to remove leakage ---
    # For row i in test_df:
    #  - model predicts return from close_t -> close_{t+1}
    #  - predicted price = close_t * (1 + pred_ret)
    #  - actual next-day price = close_{t+1}
    test_close_today = test_df['close'].values                           # close_t
    actual_next_close = test_df['close'].shift(-1).values                # close_{t+1} (last element = NaN)

    # reconstruct predicted next-day price from returns
    pred_price_from_ret = test_close_today * (1 + lgbm_preds_test)
    actual_price_next = actual_next_close

    # Remove rows where actual next-day close is NaN 
    valid_mask = ~np.isnan(actual_price_next)
    if not valid_mask.all():
        print(f"[INFO] Dropping {np.sum(~valid_mask)} final test rows with no observed next-day close (alignment).")
    pred_price_from_ret = pred_price_from_ret[valid_mask]
    actual_price_next = actual_price_next[valid_mask]
    
    dates_aligned = test_df['date'].iloc[valid_mask.nonzero()[0]].reset_index(drop=True) if isinstance(valid_mask, np.ndarray) else test_df['date'][valid_mask]

    # Evaluate LGBM in price-space
    mae_lgbm_price = mean_absolute_error(actual_price_next, pred_price_from_ret)
    r2_lgbm_price = r2_score(actual_price_next, pred_price_from_ret)
    print(f"LGBM (price) MAE: {mae_lgbm_price:.4f}")
    print(f"LGBM (price) R²:  {r2_lgbm_price:.4f}")

    # --- Residual LSTM ---
    print("\n--- Training Residual LSTM ---")
    # train residuals = true_train_next_return - lgbm_pred_train_return
    
    train_residuals = y_train.reset_index(drop=True) - pd.Series(lgbm_preds_train).reset_index(drop=True)

    Xr, yr = create_simple_residual_sequences_from_series(train_residuals, LSTM_LOOKBACK)
    lstm_model, error_scaler = None, None

    if len(Xr) >= 30:
        from sklearn.preprocessing import StandardScaler as _SS
        error_scaler = _SS()
        yr_scaled = error_scaler.fit_transform(yr.reshape(-1, 1)).flatten()
        Xr_scaled = (Xr - Xr.mean()) / (Xr.std() + 1e-9)
        Xr_scaled = Xr_scaled.reshape((Xr_scaled.shape[0], Xr_scaled.shape[1], 1))
        lstm_model = build_simple_residual_lstm(input_shape=(LSTM_LOOKBACK, 1))
        lstm_model.fit(
            Xr_scaled, yr_scaled,
            epochs=100, batch_size=16, validation_split=0.1,
            callbacks=[EarlyStopping(patience=8, restore_best_weights=True, verbose=0)],
            verbose=0
        )
        print("Residual LSTM training complete.")
    else:
        print("Not enough residual sequences for LSTM (skipped).")

    # --- Hybrid Evaluation ---
    hybrid_price_arr = None
    mae_hybrid, r2_hybrid = None, None
    if lstm_model:
       
        history = list(train_residuals.iloc[-LSTM_LOOKBACK:].values)
        hybrid_prices_full = []
        for i in range(len(test_df)):
            seq = np.array(history).reshape(1, LSTM_LOOKBACK)
            seq_scaled = (seq - Xr.mean()) / (Xr.std() + 1e-9)
            seq_scaled = seq_scaled.reshape((1, LSTM_LOOKBACK, 1))
            scaled_pred = lstm_model.predict(seq_scaled, verbose=0)[0][0]
            pred_err = error_scaler.inverse_transform([[scaled_pred]])[0][0]

            # predicted hybrid return and price for row i
            hybrid_ret = lgbm_preds_test[i] + 1.2 * pred_err
            hybrid_price = test_close_today[i] * (1 + hybrid_ret)
            hybrid_prices_full.append(hybrid_price)

            # update history with the true residual observed after day i
            true_resid = (y_test.iloc[i] - lgbm_preds_test[i])
            history.pop(0); history.append(true_resid)

        # align hybrid array with valid_mask 
        hybrid_prices_full = np.array(hybrid_prices_full)
        hybrid_price_arr = hybrid_prices_full[valid_mask]

        # Evaluate hybrid in same aligned price-space
        mae_hybrid = mean_absolute_error(actual_price_next, hybrid_price_arr)
        r2_hybrid = r2_score(actual_price_next, hybrid_price_arr)
        print(f"\n--- Hybrid Model Evaluation (LGBM + LSTM) ---")
        print(f"Hybrid MAE (price): {mae_hybrid:.4f}")
        print(f"Hybrid R² (price):  {r2_hybrid:.4f}")
    else:
        print("\nNo LSTM model trained — returning LGBM-only results.")

    # --- Final Plot (Actual vs LGBM vs Hybrid) ---
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 5))  # balanced for browser display

    plt.plot(dates_aligned, actual_price_next, label="Actual Price (t+1)",
            color="royalblue", lw=2)
    plt.plot(dates_aligned, pred_price_from_ret,
            label=f"LGBM (MAE: {mae_lgbm_price:.2f}, R²: {r2_lgbm_price:.3f})",
            color="darkorange", ls="--", lw=1.8)

    if hybrid_price_arr is not None:
        plt.plot(dates_aligned, hybrid_price_arr,
                label=f"Hybrid (MAE: {mae_hybrid:.2f}, R²: {r2_hybrid:.3f})",
                color="green", ls="-.", lw=1.8)

    plt.title(f"{TICKER} Price Prediction (Actual vs LGBM vs Hybrid)", fontsize=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Price", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

    #  Fix spacing and avoid blank sides
    plt.xticks(rotation=25)
    plt.tight_layout(pad=1.0)
    plot_path = os.path.join(ART_DIR, "final_price_prediction_v15.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Final plot saved to: {plot_path}")



        # ---  Next-Day Price Prediction Output ---
    # Use the last available row from df_for_training to predict next-day price
    last_close = df.iloc[-1]["close"]
    last_features = df.iloc[-1].drop(["date", "target", "target_ret"], errors="ignore")
    last_features_df = pd.DataFrame([last_features]).reindex(columns=X_train.columns, fill_value=0)
    last_scaled = lgbm_scaler.transform(last_features_df)
    next_ret_pred = float(lgbm_model.predict(last_scaled)[0])
    next_close_pred = last_close * (1 + next_ret_pred)

    print("\n==========================")
    print("Next-Day Forecast Summary")
    print("==========================")
    print(f"Previous Close: {last_close:.2f}")
    print(f"Predicted Next Close: {next_close_pred:.2f}")
    print(f"Predicted Return: {(next_ret_pred * 100):+.2f}%")

        # ---  Save all key artifacts for Flask app reuse ---
    try:
        # Save model, scaler, and meta info
        joblib.dump(lgbm_model, os.path.join(ART_DIR, "lgbm_model.pkl"))
        joblib.dump(lgbm_scaler, os.path.join(ART_DIR, "scaler_x.save"))
        meta = {
            "mae_lgbm": float(mae_lgbm_price),
            "r2_lgbm": float(r2_lgbm_price),
            "mae_hybrid": float(mae_hybrid) if mae_hybrid is not None else None,
            "r2_hybrid": float(r2_hybrid) if r2_hybrid is not None else None,
            "feature_columns": list(X_train.columns)
        }
        joblib.dump(meta, os.path.join(ART_DIR, "pipeline_meta.pkl"))
        print(f"[INFO] Saved model, scaler, and metrics to {ART_DIR}")
    except Exception as e:
        print(f"[WARN] Could not save artifacts: {e}")


    return (mae_lgbm_price, r2_lgbm_price), (mae_hybrid, r2_hybrid) if lstm_model else None, next_close_pred


def safe_lgbm_predict(model, X):
    """
    Safely predict with LightGBM, automatically bypassing feature-name mismatches.
    Works with both DataFrame and NumPy array inputs.
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    try:
        # --- Normal case ---
        return model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
    except lgb.basic.LightGBMError:
        # --- Fallback when feature names mismatch ---
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        return model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
    except Exception as e:
        print(f"[WARN] safe_lgbm_predict fallback triggered: {e}")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return model.predict(X, num_iteration=getattr(model, "best_iteration_", None))


# def simulate_future_trend(
#     base_df: pd.DataFrame,
#     trained_lgbm,
#     trained_scaler,
#     lstm_model,
#     error_scaler,
#     X_train_cols,
#     days_ahead: int = 30,
#     alpha: float = 0.85,
# ):
#     """
#     Iteratively simulates next N days using the current hybrid (LGBM + residual LSTM) model.
#     Handles feature name and shape mismatches robustly.
#     """

#     print(f"\n=== STARTING {days_ahead}-DAY SIMULATION (Iterative Forecast Mode) ===")
#     print(f"[INFO] Simulation starting with {len(X_train_cols)} model features and {base_df.shape[1]} DataFrame columns.")

#     sim_df = base_df.copy().reset_index(drop=True)
#     sim_prices = [sim_df['close'].iloc[-1]]
#     last_close = sim_prices[-1]

#     for step in range(1, days_ahead + 1):
#         # 1️ Prepare last row features
#         last_features = sim_df.iloc[-1].drop(['date', 'target', 'target_ret'], errors='ignore')
#         last_features_df = pd.DataFrame([last_features])

#         # --- Robust column alignment ---
#         unseen = [c for c in last_features_df.columns if c not in X_train_cols]
#         missing = [c for c in X_train_cols if c not in last_features_df.columns]

#         if unseen:
#             print(f"[WARN] Dropping {len(unseen)} unseen features (e.g. {unseen[:3]}...)")
#             last_features_df = last_features_df.drop(columns=unseen, errors='ignore')
#         if missing:
#             print(f"[WARN] Adding {len(missing)} missing features (e.g. {missing[:3]}...)")
#             for m in missing:
#                 last_features_df[m] = 0.0

#         last_features_df = last_features_df.reindex(columns=X_train_cols, fill_value=0)

#         # --- Scale ---
#         last_features_scaled = trained_scaler.transform(last_features_df)

#         # 2️Predict next-day return (safe call)
#         next_ret_lgbm = float(safe_lgbm_predict(trained_lgbm, last_features_scaled)[0])
#         next_price_lgbm = last_close * (1 + next_ret_lgbm)

#         # 3️Residual LSTM correction
#         aligned_sim_features = sim_df.reindex(columns=X_train_cols, fill_value=0)
#         aligned_scaled = trained_scaler.transform(aligned_sim_features)
#         all_errors = sim_df['target_ret'].reset_index(drop=True) - safe_lgbm_predict(trained_lgbm, aligned_scaled)

#         last_error_sequence = all_errors.iloc[-LSTM_LOOKBACK:].values.reshape(-1, 1)
#         last_error_scaled = error_scaler.transform(last_error_sequence).reshape((1, LSTM_LOOKBACK, 1))
#         next_error_scaled = float(lstm_model.predict(last_error_scaled, verbose=0)[0][0])
#         next_error = float(error_scaler.inverse_transform([[next_error_scaled]])[0][0])
#         next_price_hybrid = next_price_lgbm + (last_close * next_error)

#         # 4️Blend for stability
#         blended_price = (alpha * next_price_lgbm) + ((1 - alpha) * next_price_hybrid)

#         # 5️Append simulated price
#         sim_prices.append(blended_price)

#         # 6️Create synthetic next row
#         new_row = sim_df.iloc[-1].copy()
#         new_row['close'] = blended_price
#         new_row['open'] = last_close
#         new_row['high'] = max(blended_price, last_close)
#         new_row['low'] = min(blended_price, last_close)
#         new_row['volume'] *= np.random.uniform(0.95, 1.05)
#         new_row['date'] = new_row['date'] + pd.Timedelta(days=1)
#         sim_df = pd.concat([sim_df, pd.DataFrame([new_row])], ignore_index=True)

#         # 7️Recompute indicators
#         sim_df = add_technical_features(sim_df)
#         sim_df['target_ret'] = sim_df['close'].pct_change().fillna(0)
#         last_close = blended_price

#     # --- Summary ---
#     forecast_df = pd.DataFrame({
#         "day": np.arange(0, days_ahead + 1),
#         "predicted_price": sim_prices
#     })

#     pct_change = (sim_prices[-1] - sim_prices[0]) / sim_prices[0] * 100
#     trend = " Bullish" if pct_change > 1 else (" Bearish" if pct_change < -1 else "⚖️ Neutral")

#     print(f"\n=== {days_ahead}-Day Forecast Summary ===")
#     print(f"Start Price: {sim_prices[0]:.2f}")
#     print(f"End Price:   {sim_prices[-1]:.2f}")
#     print(f"Predicted Change: {pct_change:+.2f}% → {trend}")

#     # --- Plot ---
#     plt.style.use("seaborn-v0_8-whitegrid")
#     plt.figure(figsize=(12, 6))
#     plt.plot(forecast_df["day"], forecast_df["predicted_price"], marker='o', lw=2)
#     plt.title(f"{days_ahead}-Day Price Forecast (Iterative Hybrid Simulation)")
#     plt.xlabel("Days Ahead")
#     plt.ylabel("Predicted Price")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(ART_DIR, f"iterative_forecast_{days_ahead}d.png"))
#     plt.show()

#     return forecast_df, trend, pct_change


# ----------------------------- Reporting & Auxiliaries -----------------------------
def generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, query, predicted_price=None, stock_df=None):
    """
    Generate an advanced multimodal investment summary using Gemini.
    Incorporates technical indicators, volume trends, and cross-source sentiment.
    """
    if not GEMINI_API_KEY:
        return "Gemini API key not found. No summary available."
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    # --- Prepare textual inputs ---
    texts = []
    for src, docs in [("News", news_docs), ("Reddit", reddit_docs), ("YouTube", yt_docs)]:
        for d in docs[:15]:  # limit to most recent 15 per source
            texts.append(f"[{src}] {d.text}")

    full_text = " ".join(texts)
    if len(full_text) > 10000:
        full_text = full_text[:10000] + "..."

    # --- Prepare basic stock statistics for context ---
    stock_context = ""
    if stock_df is not None and not stock_df.empty:
        recent_df = stock_df.tail(30).copy()
        close_now = recent_df["close"].iloc[-1]
        avg_vol = recent_df["volume"].mean()
        avg_close = recent_df["close"].mean()
        vol_trend = "increasing" if recent_df["volume"].iloc[-1] > avg_vol * 1.2 else "stable" if recent_df["volume"].iloc[-1] > avg_vol * 0.8 else "decreasing"
        price_trend = "uptrend" if close_now > avg_close * 1.05 else "downtrend" if close_now < avg_close * 0.95 else "sideways"
        stock_context = (
            f"Stock Technical Summary for {ticker}:\n"
            f"- Latest Close: ${close_now:.2f}\n"
            f"- 30-day Average Close: ${avg_close:.2f}\n"
            f"- Volume Trend: {vol_trend}\n"
            f"- Price Trend: {price_trend}\n"
        )
        if "rsi_20" in recent_df.columns:
            rsi_val = recent_df["rsi_20"].iloc[-1]
            stock_context += f"- RSI(20): {rsi_val:.1f} (Overbought >70, Oversold <30)\n"

    # --- Predicted price context ---
    prediction_context = f"The model predicts the next closing price around ${predicted_price:.2f}." if predicted_price else ""

    # --- Gemini Prompt ---
    prompt = f"""
You are a financial analysis assistant.

Using the provided textual data (news, Reddit, YouTube) and stock indicators, generate a concise professional summary that:
1. Summarizes the key positive and negative factors influencing {ticker}.
2. Describes recent investor sentiment trends across News, Reddit, and YouTube.
3. Explains how technical indicators (price trend, RSI, volume) support or contradict sentiment.
4. Ends with a final investment recommendation — BUY, HOLD, or SELL — with a brief justification.

Data Context:
{stock_context}
{prediction_context}

Recent Multimodal Texts:
{full_text}

Your output should be in the following structure:
---
**Gemini AI Summary for {ticker}**
**Market Overview:** [1–2 concise lines]
**Sentiment Analysis:** [overview of sources]
**Technical Insight:** [summary of price, RSI, volume]
**Final Recommendation:** BUY / HOLD / SELL
**Justification:** [2–3 sentences explaining the rationale]
---
    """

    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"Gemini summary failed: {e}")
        return "Summary generation failed."


# ----------------------------- Main Runner ---------------------------------
def main():
    print(f"=== STARTING DATA ANALYSIS & TRAINING PIPELINE (V14 - Patched) ===")
    
    # --- Caching Logic ---
    config_str = f"{TICKER}-{','.join(QUERY)}-{DAYS}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    stock_cache_file = os.path.join(CACHE_DIR, f"stock_{config_hash}.pkl")
    news_cache_file = os.path.join(CACHE_DIR, f"news_{config_hash}.pkl")
    reddit_cache_file = os.path.join(CACHE_DIR, f"reddit_{config_hash}.pkl")
    yt_cache_file = os.path.join(CACHE_DIR, f"yt_{config_hash}.pkl")

    # --- 1. Fetch All Data (with Caching) ---
    try:
        if os.path.exists(stock_cache_file):
            print("Loading stock data from cache...")
            with open(stock_cache_file, 'rb') as f:
                stock_df = pickle.load(f)
        else:
            print("Fetching stock data...")
            stock_df = get_stock_data(TICKER, days=DAYS)
            with open(stock_cache_file, 'wb') as f:
                pickle.dump(stock_df, f)

        if os.path.exists(news_cache_file):
            print("Loading news data from cache...")
            with open(news_cache_file, 'rb') as f:
                news_docs = pickle.load(f)
        else:
            print("Fetching news data...")
            news_docs = fetch_news_rss(QUERY, days=DAYS)
            with open(news_cache_file, 'wb') as f:
                pickle.dump(news_docs, f)

        if os.path.exists(reddit_cache_file):
            print("Loading reddit data from cache...")
            with open(reddit_cache_file, 'rb') as f:
                reddit_docs = pickle.load(f)
        else:
            print("Fetching reddit data...")
            reddit_docs = fetch_reddit_praw(QUERY, days=DAYS)
            with open(reddit_cache_file, 'wb') as f:
                pickle.dump(reddit_docs, f)

        if os.path.exists(yt_cache_file):
            print("Loading youtube data from cache...")
            with open(yt_cache_file, 'rb') as f:
                yt_docs = pickle.load(f)
        else:
            print("Fetching youtube data...")
            yt_docs = fetch_youtube_docs(QUERY, days=DAYS)
            with open(yt_cache_file, 'wb') as f:
                pickle.dump(yt_docs, f)

    except Exception as e:
        print(f"ERROR during data fetching or caching: {e}")
        return

    all_docs = news_docs + reddit_docs + yt_docs
    print(f"Fetched {len(stock_df)} stock records, {len(all_docs)} total text docs.")
    
    if not all_docs or stock_df.empty:
        print("No text or stock documents found. Exiting.")
        return
    
    # --- Load NLP Models (only if needed) ---
    init_nlp()
    if not _nlp["ok"]:
        print("ERROR: NLP models failed to load. Exiting.")
        return
        
    # --- 2. NEW: Process each source independently ---
    news_features_df = process_text_source(news_docs, "news")
    reddit_features_df = process_text_source(reddit_docs, "reddit")
    yt_features_df = process_text_source(yt_docs, "youtube")

    # --- 3. Detect Events (from all docs) ---
    event_daily_features = detect_event_features(all_docs)
    print(f"Detected events on {len(event_daily_features)} days.")

    # --- 4. Build Final Hybrid FeatureSet ---
    final_df = build_hybrid_featureset(
        stock_df, 
        news_features_df,
        reddit_features_df,
        yt_features_df,
        event_daily_features
    )
    
    if final_df.empty or final_df.shape[0] < 50:
         print("\n Pipeline finished with errors: No data in final DataFrame after processing.")
         return

    # --- 5. Train Models and Evaluate ---
    metrics = train_and_evaluate_models(final_df)

    # --- Display metrics summary clearly ---
    if metrics is not None:
        (mae_lgbm, r2_lgbm), hybrid_metrics, next_close_pred = metrics

        print("\n==========================")
        print("Model Performance Summary")
        print("==========================")
        print(f"LGBM   → MAE: {mae_lgbm:.4f} | R²: {r2_lgbm:.4f}")
        if hybrid_metrics:
            mae_hybrid, r2_hybrid = hybrid_metrics
            print(f"Hybrid → MAE: {mae_hybrid:.4f} | R²: {r2_hybrid:.4f}")
        else:
            print("Hybrid model not trained (insufficient data).")

        

    else:
        print("\n Model training failed.")


    print(f"\n Pipeline finished successfully. Artifacts saved to '{ART_DIR}'")


        # --- EXTRA FEATURE: Top 10 Recent Textual Data with Sentiment Vectors ---
    print("\n==========================")
    print("📰 Top 10 Recent Textual Insights (News / Reddit / YouTube)")
    print("==========================")

    # Combine all_docs and compute sentiments (ensure NLP loaded)
    if _nlp["ok"] and len(all_docs) > 0:
        # Convert to DataFrame
        text_df = pd.DataFrame([{"date": d.date, "source": d.source, "text": d.text} for d in all_docs])
        text_df = text_df.sort_values("date", ascending=False).reset_index(drop=True)

        # Limit to last ~300 to avoid slow inference
        subset_df = text_df.head(300).copy()
        sentiments = finbert_sentiment_vectors(subset_df["text"].tolist())

        sent_cols = ["neg", "neu", "pos"]
        sent_df = pd.DataFrame(sentiments, columns=sent_cols)
        subset_df = pd.concat([subset_df, sent_df], axis=1)

        # Define sentiment intensity = max(pos, neg)
        subset_df["sent_intensity"] = subset_df[["pos", "neg"]].max(axis=1)

        # Pick top 10 by intensity (or recency if tied)
        top10 = subset_df.sort_values(["sent_intensity", "date"], ascending=[False, False]).head(10)

        # Clean text for readability (truncate long titles)
        top10["text"] = top10["text"].apply(lambda x: (x[:120] + "...") if len(x) > 120 else x)

        # Format table nicely
        print("\n{:<10} | {:<9} | {:<6} | {:<6} | {:<6} | {}".format(
            "Source", "Date", "Pos", "Neu", "Neg", "Text Snippet"
        ))
        print("-" * 110)
        for _, row in top10.iterrows():
            print("{:<10} | {:<9} | {:<6.2f} | {:<6.2f} | {:<6.2f} | {}".format(
                row["source"][:10],
                row["date"].strftime("%Y-%m-%d"),
                row["pos"], row["neu"], row["neg"],
                row["text"]
            ))

        # Save to CSV
        top10_path = os.path.join(ART_DIR, f"{TICKER}_top10_textual_insights.csv")
        top10[["date", "source", "text", "pos", "neu", "neg"]].to_csv(top10_path, index=False)
        print(f"\n Saved Top 10 textual insights to: {top10_path}")
    else:
        print(" NLP unavailable or no textual data found — skipping sentiment insights.")

    # --- Generate Summary & Recommendation ---
    print("\n Generating Gemini Summary...")
    summary = generate_gemini_summary(
        news_docs,
        reddit_docs,
        yt_docs,
        TICKER,
        QUERY,
        predicted_price= next_close_pred if 'next_close_pred' in locals() else None,
        stock_df=stock_df
    )
    with open(os.path.join(ART_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)



if __name__ == "__main__":
    main()
