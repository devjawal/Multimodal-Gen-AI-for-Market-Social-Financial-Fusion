#!/usr/bin/env python3
"""
new6_hybrid_advanced.py

Hybrid Stock Predictor with Advanced Features:
- Feature-based LightGBM model
- Residual LSTM correction for temporal patterns
- Multimodal features from stock, news, reddit, YouTube
- NEW: Event detection and embedding for key corporate events
- NEW: Long-term memory windows for technical indicators and lags
- NEW: Attention-based sentiment fusion (weighted by volatility and volume)
"""

#!/usr/bin/env python3
import os


# your existing env settings (keep these)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- CERT BUNDLE ENSURE (paste at top, before network imports) ---
try:
    import certifi, os
    ca_path = certifi.where()
    # Make sure all HTTP clients use the same CA bundle
    os.environ['REQUESTS_CA_BUNDLE'] = ca_path
    os.environ['SSL_CERT_FILE'] = ca_path
    os.environ['CURL_CA_BUNDLE'] = ca_path
    print(f"[DEBUG] Using certifi CA bundle at: {ca_path}")
except Exception as _e:
    print(f"[WARN] certifi import failed or not installed: {_e}")
# --- END ---





import joblib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import numpy as np
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, early_stopping

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _YT_TRANSCRIPT_AVAILABLE = True
except Exception:
    _YT_TRANSCRIPT_AVAILABLE = False

from dotenv import load_dotenv
import praw
import yfinance as yf
from dateutil import parser

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- User Configuration ---
TICKER = "AAPL"
QUERY = ["iPhone", "Tim Cook", "MacBook", "Apple", "Apple INC", "iphone 17"]
DAYS = 1200
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Environment & Paths ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "stock-predictor-v1 by user")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

ART_DIR = os.path.join(os.getcwd(), "artifacts_lgbm_advanced")
os.makedirs(ART_DIR, exist_ok=True)

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
EMB_DIM = 384  # Dimension for 'all-MiniLM-L6-v2'

def init_nlp():
    if _nlp["ok"]:
        return
    print("Loading NLP models...")
    try:
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
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
    # Process in chunks to avoid overwhelming the pipeline
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
            # If a chunk fails, append neutral scores for that chunk size
            results.extend([np.array([0.0, 1.0, 0.0])] * len(text_chunk))
    return np.array(results)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts or not _nlp["ok"]:
        return np.zeros((len(texts), EMB_DIM))
    return _nlp["embedder"].encode(texts, show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True, batch_size=32)

# ----------------------------- Data Fetching ---------------------------------
def get_stock_data(ticker: str, days: int) -> pd.DataFrame:
    """Fetches stock data and ensures clean, single-level, lowercase column names."""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No stock data for {ticker}")

    df = df.reset_index()

    # --- START: CORRECTED CODE BLOCK ---
    
    # Robustly handle MultiIndex (tuple) columns and flatten them
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            name = '_'.join(filter(None, col)) # e.g., ('Close', '') -> 'Close'
        else:
            name = col
        new_columns.append(name.lower().replace(" ", "_"))
    df.columns = new_columns

    # Use the robust renaming map from your reference code
    # This ensures 'adj_close' or 'close_aapl' etc. become 'close'
    rename_map = {
        "adj_close": "close",
        f"close_{ticker.lower()}": "close",
        f"open_{ticker.lower()}": "open",
        f"high_{ticker.lower()}": "high",
        f"low_{ticker.lower()}": "low",
        f"volume_{ticker.lower()}": "volume",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Ensure the 'date' column exists
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df.rename(columns={c: "date"}, inplace=True)
                break
        else: # If loop finishes without finding a date column
             raise RuntimeError(f"'date' column missing! Columns: {df.columns.tolist()}")

    # Add a final, crucial check to ensure the 'close' column now exists
    if "close" not in df.columns:
        raise RuntimeError(f"'close' column missing after renaming! Columns: {df.columns.tolist()}")
    
    # --- END: CORRECTED CODE BLOCK ---

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)

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
                text = clean_text(f"{getattr(entry, 'title', '')}. {getattr(entry, 'summary', '')}")
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
            for submission in reddit.subreddit(subs).search(query, sort="new", limit=300):
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                if created < cutoff: continue
                text = clean_text(f"{submission.title}. {submission.selftext or ''}")
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
                transcript = ""
                if _YT_TRANSCRIPT_AVAILABLE:
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                        transcript = " ".join([t["text"] for t in transcript_list])
                    except Exception: pass
                text = clean_text(f"{title}. {transcript}")
                if text:
                    docs.append(Doc(date=pd.to_datetime(published_at).normalize(), text=text, source="youtube"))
                    seen.add(video_id)
        except Exception as e:
            print(f"WARNING: YouTube fetch failed for query '{query}': {e}")
    return docs

# ----------------------------- Feature Engineering ---------------------------------
def aggregate_text_features(docs: List[Doc], source: str) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()
    df_initial = pd.DataFrame([{"date": d.date, "text": d.text} for d in docs])
    sentiments = finbert_sentiment_vectors(df_initial["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=[f"{source}_sent_neg", f"{source}_sent_neu", f"{source}_sent_pos"])
    embeddings = embed_texts(df_initial["text"].tolist())
    emb_cols = [f"{source}_emb_{i}" for i in range(EMB_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    df = pd.concat([df_initial.reset_index(drop=True), sent_df, emb_df], axis=1)
    agg_funcs = {col: "mean" for col in df.columns if col not in ["date", "text"]}
    daily_df = df.groupby("date").agg(agg_funcs).reset_index()
    return daily_df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """MODIFIED: Added long-term memory windows."""
    out = df.copy()
    close = out["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    # Added longer windows: 60, 120 days for hierarchical memory
    for win in [5, 10, 20, 60, 120]:
        out[f"sma_{win}"] = close.rolling(win).mean()
        # RSI calculation robust to division by zero
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(win).mean()
        loss = -delta.where(delta < 0, 0).rolling(win).mean()
        rs = gain / (loss + 1e-9)
        out[f"rsi_{win}"] = 100 - (100 / (1 + rs))
        out[f"vol_{win}"] = out["ret_1"].rolling(win).std()
    return out

def detect_event_features(all_docs: List[Doc]) -> pd.DataFrame:
    """NEW: Detects corporate events from text and creates event-specific features."""
    if not all_docs:
        return pd.DataFrame()

    EVENT_KEYWORDS: Dict[str, List[str]] = {
        "earnings": ["earnings call", "quarterly results", "revenue", "profit", "guidance"],
        "launch": ["product launch", "announcement", "release date", "unveiled", "new iphone"],
        "legal": ["lawsuit", "investigation", "sec", "doj", "settlement", "antitrust"],
        "partnership": ["partnership", "collaboration", "acquisition", "merger", "deal"],
    }

    event_data = []
    for doc in all_docs:
        event_texts = []
        is_event = False
        for event_type, keywords in EVENT_KEYWORDS.items():
            # Use regex to find whole words, case-insensitive
            if any(re.search(r'\b' + kw + r'\b', doc.text, re.IGNORECASE) for kw in keywords):
                event_texts.append(doc.text)
                is_event = True
        if is_event:
            event_data.append({"date": doc.date, "text": ". ".join(event_texts)})

    if not event_data:
        return pd.DataFrame()

    df = pd.DataFrame(event_data)
    df = df.drop_duplicates(subset=["date", "text"])
    sentiments = finbert_sentiment_vectors(df["text"].tolist())
    sent_df = pd.DataFrame(sentiments, columns=["event_sent_neg", "event_sent_neu", "event_sent_pos"])
    df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

    # Aggregate by day
    agg_funcs = {col: "mean" for col in sent_df.columns}
    daily_df = df.groupby("date").agg(agg_funcs).reset_index()
    daily_df["is_event_day"] = 1
    return daily_df



def build_hybrid_featureset(stock_df, news_df, reddit_df, yt_df, event_df) -> pd.DataFrame:
    """MODIFIED: Integrates event, long-term memory, and attention features."""
    print("Building hybrid feature set with advanced features...")
    features = stock_df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)

    # Merge all data sources
    for text_df in [news_df, reddit_df, yt_df, event_df]:
        if not text_df.empty:
            text_df["date"] = pd.to_datetime(text_df["date"]).dt.tz_localize(None)
            features = pd.merge(features, text_df, on="date", how="left")

    # Add technical indicators (now with long-term memory)
    features = add_technical_features(features)

    # --- NEW: Attention-based Fusion ---
    # Scale volatility and volume to be used as weights (0-1 range)
    scaler = MinMaxScaler()
    features['vol_20_scaled'] = scaler.fit_transform(features[['vol_20']].fillna(0))
    features['volume_scaled'] = scaler.fit_transform(features[['volume']].fillna(0))

    # Calculate composite sentiment scores
    news_sentiment_score = features.get('news_sent_pos', 0) - features.get('news_sent_neg', 0)
    reddit_sentiment_score = features.get('reddit_sent_pos', 0) - features.get('reddit_sent_neg', 0)

    # Create attention features
    features['news_attention_score'] = news_sentiment_score * features['vol_20_scaled']
    features['reddit_attention_score'] = reddit_sentiment_score * features['volume_scaled']
    print("Attention features created.")

    # --- MODIFIED: Long-term Memory Lags ---
    lag_periods = [1, 2, 3, 5, 10, 21, 63] # Added ~1 month and ~1 quarter lags
    lag_cols = [col for col in features.columns if col not in ['date', 'open', 'high', 'low', 'close']]
    # Exclude the scaled columns from lagging as they are used for current day's attention
    lag_cols = [c for c in lag_cols if '_scaled' not in c]

    for lag in lag_periods:
        shifted = features[lag_cols].shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in shifted.columns]
        features = pd.concat([features, shifted], axis=1)

    features['target'] = features['close'].shift(-1)
    features = features.dropna(subset=["target", "close"])
    features = features.fillna(0)
    print(f"Final feature set shape: {features.shape}")
    return features.reset_index(drop=True)


# ----------------------------- Residual LSTM ---------------------------------
def build_residual_sequences(y_true, y_pred, lookback=10):
    residuals = y_true - y_pred
    X, y = [], []
    for i in range(len(residuals) - lookback):
        X.append(residuals.iloc[i:i+lookback].values)
        y.append(residuals.iloc[i+lookback])
    return np.array(X), np.array(y)

def train_residual_lstm(y_true, y_pred, lookback=10):
    X, y = build_residual_sequences(y_true, y_pred, lookback)
    if len(X) < 2: # Need at least 2 samples to train
        print("⚠️ Not enough data for residual LSTM.")
        return None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0) # Verbose set to 0
    print("Residual LSTM training complete.")
    return model

# ----------------------------- Reporting & Auxiliaries -----------------------------
def generate_gemini_summary(news_docs, reddit_docs, yt_docs, ticker, query, predicted_price=None):
    """Generate concise multimodal summary using Gemini."""
    if not GEMINI_API_KEY:
        return "Gemini API key not found. No summary available."
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    texts = []
    for src, docs in [("News", news_docs), ("Reddit", reddit_docs), ("YouTube", yt_docs)]:
        for d in docs[:15]: # Limit docs per source
             texts.append(f"[{src}] {d.text}")

    extra = f"\nThe model predicts the next closing price will be around ${predicted_price:.2f}." if predicted_price else ""
    # Truncate total text to avoid API limits
    full_text = " ".join(texts)
    max_len = 10000 # Safety limit
    truncated_text = full_text[:max_len] + ('...' if len(full_text) > max_len else '')

    prompt = f"""
    Summarize the following recent financial signals for {ticker} ({','.join(query)}).
    Focus on:
    1. Key market-moving events or topics (e.g., earnings, product news).
    2. Overall investor sentiment (bullish, bearish, neutral) from each source.
    3. Any conflicting signals between sources.
    
    Keep the summary concise and professional.
    
    Recent Texts:
    {truncated_text}
    {extra}
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"Gemini summary failed: {e}")
        return "Summary generation failed."

# ----------------------------- Training & Evaluation ---------------------------------
def train_and_evaluate_lgbm(df: pd.DataFrame):
    if df.shape[0] < 50:
        print("ERROR: Insufficient data for training after feature engineering.")
        return None

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    y_train = train_df['target']
    X_train = train_df.drop(columns=['date', 'target', 'close'])
    y_test = test_df['target']
    X_test = test_df.drop(columns=['date', 'target', 'close'])

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Ensure column order is identical
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        random_state=RANDOM_SEED, n_jobs=-1, colsample_bytree=0.8, subsample=0.8
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)], eval_metric='l1',
        callbacks=[early_stopping(100, verbose=False)]
    )

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\n--- Model Evaluation (LGBM) ---")
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    joblib.dump(model, os.path.join(ART_DIR, "lgbm_model.pkl"))
    joblib.dump(scaler, os.path.join(ART_DIR, "scaler.pkl"))

        # --- Save metadata for exact parity with the web app ---
    try:
        # feature names MUST be exactly the columns used to train the model (order matters)
        feature_names = list(X_train.columns)
        meta = {
            "feature_names": feature_names,
            "lookback": 10,  # same lookback used for residual LSTM
            "mae_train": float(mae),
            "r2_train": float(r2),
            "training_date": datetime.utcnow().isoformat(),
            "days": DAYS
        }
        joblib.dump(meta, os.path.join(ART_DIR, "pipeline_meta.pkl"))
        print(f"Saved pipeline metadata to {os.path.join(ART_DIR, 'pipeline_meta.pkl')}")
    except Exception as e:
        print("Warning: failed to save pipeline metadata:", e)


    # --- Residual LSTM Correction ---
    print("\nTraining residual LSTM on prediction errors...")
    residual_lstm = train_residual_lstm(y_test.reset_index(drop=True), pd.Series(predictions), lookback=10)
    
    final_predictions = predictions # Use LGBM preds as default
    
    if residual_lstm:
        lookback = 10
        X_resid, _ = build_residual_sequences(y_test.reset_index(drop=True), pd.Series(predictions), lookback=lookback)
        if len(X_resid) > 0:
            X_resid = X_resid.reshape((X_resid.shape[0], X_resid.shape[1], 1))
            residual_preds = residual_lstm.predict(X_resid).flatten()
            
            # Pad residual preds to align with original predictions
            padded_residual_preds = np.zeros_like(predictions)
            padded_residual_preds[lookback:] = residual_preds
            
            corrected_preds = predictions + padded_residual_preds
            final_predictions = corrected_preds # Update with hybrid preds

            mae_corr = mean_absolute_error(y_test, corrected_preds)
            r2_corr = r2_score(y_test, corrected_preds)
            print(f"\n--- Hybrid Model Evaluation (LGBM + Residual LSTM) ---")
            print(f"Corrected MAE: {mae_corr:.4f}")
            print(f"Corrected R²:  {r2_corr:.4f}")
            
            residual_lstm.save(os.path.join(ART_DIR, "residual_lstm.keras"))

            # --- Plotting ---
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(15, 7))
            plt.plot(test_df['date'], y_test, label="Actual Price", color="royalblue", lw=2)
            plt.plot(test_df['date'], predictions, label="LGBM Prediction", color="darkorange", ls="--")
            plt.plot(test_df['date'], corrected_preds, label="Hybrid (LGBM + LSTM)", color="green", ls="-.")
            plt.title(f"{TICKER} Price Prediction: LGBM vs Hybrid", fontsize=16)
            plt.xlabel("Date"); plt.ylabel("Price")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(ART_DIR, "prediction_comparison.png"))
            plt.show()

    last_pred = float(final_predictions[-1])
    print(f"\n🔮 Next-day Predicted Price (Hybrid): {last_pred:.2f}")
    return last_pred

# ----------------------------- Main Pipeline ---------------------------------
def main():
    print("=== STARTING HYBRID LGBM + LSTM PIPELINE (ADVANCED FEATURES) ===")
    init_nlp()

    # --- Fetch Data ---
    stock_df = get_stock_data(TICKER, days=DAYS)
    news_docs = fetch_news_rss(QUERY, days=DAYS)
    reddit_docs = fetch_reddit_praw(QUERY, days=DAYS)
    yt_docs = fetch_youtube_docs(QUERY, days=DAYS)
    all_docs = news_docs + reddit_docs + yt_docs
    print(f"Fetched {len(stock_df)} stock records, {len(news_docs)} news, {len(reddit_docs)} reddit, {len(yt_docs)} yt.")

    # --- Build Features ---
    news_daily_features = aggregate_text_features(news_docs, "news")
    reddit_daily_features = aggregate_text_features(reddit_docs, "reddit")
    yt_daily_features = aggregate_text_features(yt_docs, "yt")
    
    # NEW: Run event detection
    event_daily_features = detect_event_features(all_docs)
    print(f"Detected events on {len(event_daily_features)} days.")

    final_df = build_hybrid_featureset(
        stock_df, news_daily_features, reddit_daily_features,
        yt_daily_features, event_daily_features
    )

    if final_df.empty:
        print("\n❌ Pipeline finished with errors: No data in final DataFrame.")
        return

    predicted_price = train_and_evaluate_lgbm(final_df)
    if predicted_price is None:
        print("\n❌ Training failed.")
        return

    # --- Generate Summary & Recommendation ---
    print("\n📝 Generating Gemini Summary...")
    summary = generate_gemini_summary(news_docs, reddit_docs, yt_docs, TICKER, QUERY, predicted_price)
    with open(os.path.join(ART_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)

    print(f"\n✅ Pipeline finished successfully. Artifacts saved to '{ART_DIR}'")


if __name__ == "__main__":
    main()