# Multimodal Hybrid Framework for Stock Price Prediction: Market, Social & Financial Fusion

This project is a full-stack web application that provides short-horizon stock price predictions by fusing quantitative market data with qualitative public sentiment.

The core problem it solves is the noise and unreliability of relying on a single data source. A stock's price is influenced not only by its past performance but also by news events, social media hype, and expert analysis. This tool attempts to capture this complex, multimodal reality.

A user can enter a stock ticker (e.g., "AAPL") and receive:
* **A Next-Day Price Prediction:** A specific dollar value for the next closing price.
* **A "Buy/Sell/Hold" Signal:** An advanced, rule-based recommendation.
* **A Generative AI Summary:** A human-readable analysis, powered by Google Gemini, that explains the key drivers (news, sentiment, conflicts) behind the prediction.

---

## How It Works: The Pipeline

The system is an end-to-end pipeline that automatically executes six main stages:

### 1. Data Ingestion
The system collects data from four distinct modalities:
* **Market Data:** Historical OHLCV (Open, High, Low, Close, Volume) data is fetched from `yfinance`.
* **News Data:** Financial news articles and headlines are scraped from Google News RSS feeds using `feedparser`.
* **Social Media Data:** Posts and comments are sourced from financial subreddits (like `r/wallstreetbets` and `r/stocks`) using `praw`.
* **Video Data:** Transcripts from financial analysis videos are collected using the `youtube-transcript-api`.

### 2. NLP Preprocessing
All textual data is cleaned (`BeautifulSoup`) and then processed by two separate NLP models:
* **Sentiment Analysis:** The `ProsusAI/finbert` model is used to generate a 3-dimensional (positive, negative, neutral) sentiment vector for each document.
* **Embeddings:** The `sentence-transformers/all-MiniLM-L6-v2` model is used to generate 384-dimensional vector embeddings to capture the semantic meaning of the text.

### 3. Feature Engineering
This stage combines all data sources.
* **Technical Indicators:** Standard indicators like SMA, RSI, and volatility are calculated for multiple windows (5, 10, 20, 60, 120 days).
* **Event Detection:** Text is scanned for keywords (e.g., "earnings," "launch," "lawsuit") to create an "is\_event\_day" feature.
* **Lag Features:** Historical features are lagged (shifted) by 1, 2, 3, 5, 10, 21, and 63 days to provide the model with long-term memory.

### 4. Volatility-Weighted Attention
This is a novel fusion mechanism to reduce noise. Instead of treating all sentiment as equal, the system weights it by market conditions. A final `news_attention_score` is calculated by multiplying the news sentiment by the 20-day volatility (`vol_20_scaled`). This means sentiment on a high-volatility, high-activity day is given more importance than a random tweet on a quiet day.

### 5. Hybrid "Predictor-Corrector" Modeling
The prediction is a two-stage process:
1.  **Base Predictor (LightGBM):** A `LGBMRegressor` model is first trained on the entire fused feature set to make an initial prediction.
2.  **Error Corrector (Residual LSTM):** A `Residual LSTM` (a sequential deep learning model) is then trained on the *errors* (residuals) of the LightGBM model. It learns the temporal patterns in the errors that the tree model missed.

The **Final Prediction** is the sum of the LightGBM prediction and the Residual LSTM's error correction (`LGBM_prediction + LSTM_correction`).

### 6. Summarization & Deployment
The entire pipeline is wrapped in a Flask web application (`app.py`). The final price, signal, and raw text signals are passed to the Google Gemini API (`models/gemini-pro-latest`) to generate the final, human-readable summary for the user.

---

## 👥 Authors
* **Somya Singh Parmar** (22BCE0647)
* **Shambhavi Shree** (22BCE2531)
* **Devkaran Jawal** (22BCE3048)