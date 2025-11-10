# app2.py — Robust Flask wrapper for final1 pipeline (improved feature alignment + scaler handling)
import os
import sys
import io
import time
import joblib
import shutil
import traceback
import importlib.util
from contextlib import redirect_stdout
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash

# Use Agg backend for matplotlib (server-safe)
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
load_dotenv()

# --------- Paths ----------
HOME = os.path.expanduser("~")
CANDIDATE_ARTS = [
    os.path.join(HOME, "Desktop", "tempMultimodal", "artifacts"),
    os.path.join(HOME, "Desktop", "TEMPMULTIMODAL", "artifacts"),
    os.path.join(os.getcwd(), "artifacts"),
    os.path.join(os.getcwd(), "artifacts_lgbm_advanced"),
]
ART_DIR = next((p for p in CANDIDATE_ARTS if os.path.exists(p)), CANDIDATE_ARTS[0])
os.makedirs(ART_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL1_PATH = os.path.join(PROJECT_ROOT, "final1.py")

# --- Import final1 pipeline dynamically (if available) ---
pipeline = None
import_error = None
try:
    spec = importlib.util.spec_from_file_location("final1", FINAL1_PATH)
    final1 = importlib.util.module_from_spec(spec)
    sys.modules["final1"] = final1
    spec.loader.exec_module(final1)
    pipeline = final1
except Exception:
    import_error = traceback.format_exc()

# Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me-please")

# Create template/static dirs if missing
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
IMG_DIR = os.path.join(STATIC_DIR, "img")
for d in (TEMPLATES_DIR, STATIC_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

# --- Templates & CSS (kept small & same style as prior) ---
_layout = """<!doctype html> ... (omitted here for brevity; same as your existing layout) ... """
# To avoid repeating long template strings in this message, in your copy use the same templates you already have.
# The rest of the file assumes templates exist at templates/layout.html, templates/index.html, templates/result.html,
# and style at static/style.css. If not present, previous app2.py wrote them; keep those.

# ------------------ Utilities & robust model discovery --------------------

def find_saved_artifacts():
    """
    Return dict with located paths for model/scaler/meta.
    Tries several common filenames that you mentioned exist in your project.
    """
    candidates = {
        "model": [
            "lgbm_model.pkl", "lgbm.pkl", "model_lgbm.pkl",
            "best_stock_model.h5", "best_stock_model.pkl", "best_stock_model.joblib",
            "best_stock_model_lgbm.pkl"
        ],
        "scaler": ["scaler.pkl", "scaler_x.save", "scaler_x.pkl", "scaler_x.joblib"],
        "meta": ["pipeline_meta.pkl", "pipeline_meta.joblib", "meta.pkl", "meta.joblib"]
    }
    found = {}
    for k, names in candidates.items():
        for n in names:
            p = os.path.join(ART_DIR, n)
            if os.path.exists(p):
                found[k] = p
                break
    return found

def safe_load_model(path):
    """
    Load a model from common formats:
    - joblib/pickle saved sklearn/LightGBM models
    - keras .h5 models
    Returns (model_obj, kind) where kind in {"sklearn", "keras", "unknown"}
    """
    if not path:
        return None, None
    try:
        # keras h5
        if path.endswith(".h5") or path.endswith(".keras"):
            from tensorflow.keras.models import load_model
            m = load_model(path)
            return m, "keras"
        # try joblib
        try:
            m = joblib.load(path)
            return m, "sklearn"
        except Exception:
            # fallback pickle
            import pickle
            with open(path, "rb") as f:
                m = pickle.load(f)
            return m, "sklearn"
    except Exception:
        return None, None

def safe_load_scaler(path):
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def copy_plot_to_static(plot_basename="final_price_prediction_v15.png"):
    src = os.path.join(ART_DIR, plot_basename)
    if not os.path.exists(src):
        return None
    dst = os.path.join(IMG_DIR, plot_basename)
    try:
        shutil.copyfile(src, dst)
        return "/static/img/" + plot_basename
    except Exception:
        return None

# robust reindex-in-one-step helper (avoids fragmentation warning)
def align_features_to_expected(X_df, expected_cols):
    """
    Ensure X_df contains all expected_cols (in that order) by adding missing columns in one step.
    Returns a DataFrame ordered as expected_cols.
    """
    import pandas as pd
    missing = [c for c in expected_cols if c not in X_df.columns]
    if missing:
        add = {c: 0.0 for c in missing}
        # add all at once
        X_df = pd.concat([X_df, pd.DataFrame({k:[v]*len(X_df) for k,v in add.items()})], axis=1)
    # Reindex to exact order
    X_df = X_df.reindex(columns=expected_cols, fill_value=0.0)
    return X_df

# compute recommendation (unchanged logic; caps are unchanged)
def compute_recommendation(pred_ret, textual_top10):
    if pred_ret is None:
        return "HOLD", "No prediction available", "hold"
    sent = 0.0
    for r in textual_top10 or []:
        try:
            sent += (float(r.get("pos", 0)) - float(r.get("neg", 0)))
        except Exception:
            pass
    if textual_top10:
        sent = sent / max(1.0, len(textual_top10))
    if pred_ret > 0.02:
        return "STRONG BUY", f"Model +{pred_ret*100:.2f}% | Sent {sent:+.3f}", "buy"
    if pred_ret > 0.005:
        if sent > 0.02:
            return "BUY", f"Slight +{pred_ret*100:.2f}% confirmed by sentiment", "buy"
        if sent < -0.05:
            return "HOLD", f"Slightly positive model but negative sentiment ({sent:+.2f})", "hold"
        return "BUY", f"Model +{pred_ret*100:.2f}%", "buy"
    if pred_ret < -0.02:
        return "STRONG SELL", f"Model {pred_ret*100:.2f}% | Sent {sent:+.3f}", "sell"
    if pred_ret < -0.005:
        if sent < -0.02:
            return "SELL", f"Slight negative model (-{abs(pred_ret)*100:.2f}%) with negative sentiment", "sell"
        if sent > 0.05:
            return "HOLD", "Slight negative model but positive sentiment", "hold"
        return "SELL", f"Model -{abs(pred_ret)*100:.2f}%", "sell"
    return "HOLD", "Model prediction close to neutral", "hold"

# ------------------ Routes ---------------------

@app.route("/", methods=["GET"])
def index():
    if import_error:
        flash("Warning: failed to import final1 module; reuse helpers may be limited.", "error")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker", "AAPL").strip().upper()
    days = int(request.form.get("days", 1200))
    predict_date = request.form.get("predict_date") or None
    queries_raw = request.form.get("queries", "")
    queries = [q.strip() for q in queries_raw.split(",") if q.strip()] or [ticker]
    mode = request.form.get("mode", "reuse")

    prev_close = None
    pred_price = None
    pred_ret = None
    metrics = {"lgbm_mae": None, "lgbm_r2": None, "hybrid_mae": None, "hybrid_r2": None}
    top10_rows = []
    


    plot_url = None

    try:
        # find artifacts
        arts = find_saved_artifacts()
        model_path = arts.get("model")
        scaler_path = arts.get("scaler")
        meta_path = arts.get("meta")

        # If user chose full training, run final1.main() (it will write artifacts)
        if mode == "train":
            if not os.path.exists(FINAL1_PATH):
                flash("final1.py not found for full retrain.", "error")
                return redirect(url_for("index"))
            # call final1.main() capturing stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    # set args in module if available
                    if pipeline:
                        if hasattr(pipeline, "TICKER"):
                            pipeline.TICKER = ticker
                        if hasattr(pipeline, "QUERY"):
                            pipeline.QUERY = queries
                        if hasattr(pipeline, "DAYS"):
                            pipeline.DAYS = int(days)
                        pipeline.main()
                    else:
                        # dynamic import fallback
                        spec2 = importlib.util.spec_from_file_location("final1", FINAL1_PATH)
                        mod = importlib.util.module_from_spec(spec2)
                        sys.modules["final1"] = mod
                        spec2.loader.exec_module(mod)
                        if hasattr(mod, "TICKER"):
                            mod.TICKER = ticker
                        if hasattr(mod, "QUERY"):
                            mod.QUERY = queries
                        if hasattr(mod, "DAYS"):
                            mod.DAYS = int(days)
                        mod.main()
                except Exception:
                    traceback.print_exc(file=buf)
            # after training, refresh artifact paths
            arts = find_saved_artifacts()
            model_path = arts.get("model")
            scaler_path = arts.get("scaler")
            meta_path = arts.get("meta")

        # --- Build features using pipeline helpers (required) ---
        if pipeline is None:
            flash("Pipeline helpers (final1) not importable — cannot build features.", "error")
            return redirect(url_for("index"))

        stock_df = pipeline.get_stock_data(ticker, days=days)
        news_docs = pipeline.fetch_news_rss(queries, days=days)
        reddit_docs = pipeline.fetch_reddit_praw(queries, days=days)
        yt_docs = pipeline.fetch_youtube_docs(queries, days=days)
        event_df = pipeline.detect_event_features(news_docs + reddit_docs + yt_docs)
        news_feats = pipeline.process_text_source(news_docs, "news")
        reddit_feats = pipeline.process_text_source(reddit_docs, "reddit")
        yt_feats = pipeline.process_text_source(yt_docs, "youtube")
        final_df = pipeline.build_hybrid_featureset(stock_df, news_feats, reddit_feats, yt_feats, event_df)

        if final_df.empty:
            flash("Feature engineering returned empty DataFrame — aborting.", "error")
            return redirect(url_for("index"))

        

        # --- Choose the row (date) for prediction ---
        import pandas as pd

        if predict_date:
            try:
                # Convert string to datetime for exact/nearest matching
                predict_date = pd.to_datetime(predict_date)

                # If exact date available
                row = final_df[final_df["date"] == predict_date]

                # If no exact match, use the closest available date
                if row.empty:
                    closest_idx = (final_df["date"] - predict_date).abs().argsort()[:1]
                    row = final_df.iloc[closest_idx]
                    print(f"[INFO] No exact match for {predict_date.date()}, using closest date: {row['date'].iloc[0].date()}")
            except Exception as e:
                print(f"[WARN] Could not match date properly: {e}. Using latest row instead.")
                row = final_df.tail(1)
        else:
            # Default: predict for the latest available data
            row = final_df.tail(1)

        row = row.reset_index(drop=True)
        prev_close = float(row["close"].iloc[0])


        # Drop non-feature cols
        X = row.drop(columns=["date", "target_ret", "target"], errors="ignore")

        # --- Load model, scaler, and meta ---
        if not model_path:
            flash("No trained model found. Run full retrain first.", "error")
            return redirect(url_for("index"))

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
        meta = joblib.load(meta_path) if meta_path and os.path.exists(meta_path) else {}

# --- Align columns strictly to training set structure ---
        expected_cols = None
        if meta and isinstance(meta, dict):
            expected_cols = meta.get("feature_columns")
        
        if expected_cols is None and hasattr(model, "feature_name_"):
            # Fallback if meta.pkl is missing feature_columns
            expected_cols = list(model.feature_name_)

        import pandas as pd
        if expected_cols:
            print(f"[INFO] Aligning {len(X.columns)} current features to {len(expected_cols)} expected training features.")
            
            # This single reindex step handles all cases:
            # 1. Drops columns in X that are not in expected_cols.
            # 2. Adds columns from expected_cols that are not in X (fills with 0.0).
            # 3. Sorts all columns to match the exact order of expected_cols.
            X = X.reindex(columns=expected_cols, fill_value=0.0)
            
            print(f"[INFO] DataFrame aligned. Final shape for scaling: {X.shape}")

        elif scaler is not None and hasattr(scaler, "n_features_in_"):
            # Secondary fallback if we have scaler but no meta
            expected_num_features = scaler.n_features_in_
            if X.shape[1] != expected_num_features:
                flash(f"Feature mismatch: Model expects {expected_num_features} features, but pipeline generated {X.shape[1]}.", "error")
                return redirect(url_for("index"))
            print("[WARN] No 'feature_columns' list found. Proceeding with raw values; trusting column order.")
        
        elif not expected_cols:
             print("[WARN] No 'feature_columns' metadata found. Prediction may be unreliable.")


        # Ensure numeric dtype
        X = X.astype(float).fillna(0.0)


        # --- Scale input (robust to feature-name mismatches) ---
        if scaler is not None:
            try:
                # The .reindex() above ensures feature names and order match.
                # This should now work without the fallback.
                X_scaled = scaler.transform(X)
                
            except Exception as e:
                # This fallback should no longer be needed, but kept as a safety net
                import numpy as np
                print(f"[WARN] Scaler transform failed despite alignment. Using raw numpy transform. Details: {e}")
                
                # Check if the error is about number of features
                if "features" in str(e) and hasattr(scaler, "n_features_in_"):
                    n_expected = scaler.n_features_in_
                    if X.shape[1] != n_expected:
                        flash(f"CRITICAL: Scaler expects {n_expected} features, but got {X.shape[1]}. Retrain model.", "error")
                        return redirect(url_for("index"))
                
                X_scaled = scaler.transform(np.asarray(X))
        else:
            X_scaled = X.values # Use .values if no scaler

        # --- Predict next-day return ---
        try:
            pred_ret = float(model.predict(X_scaled)[0])
        except Exception as e:
            flash(f"Prediction failed: {e}", "error")
            pred_ret = None

        # --- Compute next-day predicted price (raw) ---
        pred_price_raw = prev_close * (1.0 + pred_ret) if pred_ret is not None else None

        # --- SHORT-TERM SENTIMENT ADJUSTMENT (past 7 days) ---
        # Heuristic: aggregate past week textual sentiment (news/reddit/yt),
        # then nudge pred_price by a fraction of prior day's (high-low) depending on sentiment sign.
        try:
            import pandas as pd
            lookback_days = 7
            # date of the chosen row
            chosen_date = pd.to_datetime(row["date"].iloc[0])

            # select past week (strictly before or up to chosen_date)
            mask = (final_df["date"] <= chosen_date) & (final_df["date"] > (chosen_date - pd.Timedelta(days=lookback_days)))
            past_week = final_df.loc[mask]

            # Find per-source aggregated sentiment columns if present (created in build_hybrid_featureset)
            sent_cols = [c for c in final_df.columns if c.endswith("_agg_sent")]
            # If att-style or pca-only present, fallback to pos-neg difference:
            if not sent_cols:
                pos_cols = [c for c in final_df.columns if "_sent_pos" in c]
                neg_cols = [c for c in final_df.columns if "_sent_neg" in c]
                # build synthetic agg_sent columns in-memory
                for p, n in zip(pos_cols, neg_cols):
                    # create a name that won't be stored back to final_df
                    final_df[f"_tmp_{p}_agg"] = final_df[p] - final_df[n]
                sent_cols = [c for c in final_df.columns if c.startswith("_tmp_") and c.endswith("_agg")]

            if len(past_week) == 0 or len(sent_cols) == 0:
                mean_sent = 0.0
            else:
                # mean across days then mean across sources (gives one scalar)
                mean_sent = past_week[sent_cols].mean().mean()

            # Compute prior-day high-low (use the row chosen)
            try:
                prev_hl = float(row["high"].iloc[0]) - float(row["low"].iloc[0])
                if prev_hl < 0: prev_hl = abs(prev_hl)
            except Exception:
                prev_hl = 0.0

            # Tuning knobs (change these as you experiment)
            POS_THRESH = 0.02    # above this -> positive
            NEG_THRESH = -0.02   # below this -> negative
            SCALE = 2.0          # scales how much sentiment magnitude maps to fraction of high-low
            MAX_FRACTION = 0.8   # don't allow adjustment larger than 80% of prev_hl

            # Determine adjustment fraction from mean_sent (clamped)
            adj_fraction = min(MAX_FRACTION, abs(mean_sent) * SCALE)
            # If sentiment is tiny (near neutral), shrink fraction further
            if abs(mean_sent) < 0.005:
                adj_fraction *= 0.25

            pred_price = pred_price_raw
            # Apply rule: positive -> add prev_hl*adj_fraction; negative -> subtract
            if pred_price is not None and prev_hl > 0:
                if mean_sent >= POS_THRESH:
                    pred_price = pred_price_raw + (prev_hl * adj_fraction)
                    adj_reason = f"positive_sentiment (mean={mean_sent:.3f}) -> +{prev_hl*adj_fraction:.4f}"
                elif mean_sent <= NEG_THRESH:
                    pred_price = pred_price_raw - (prev_hl * adj_fraction)
                    adj_reason = f"negative_sentiment (mean={mean_sent:.3f}) -> -{prev_hl*adj_fraction:.4f}"
                else:
                    adj_reason = f"neutral_sentiment (mean={mean_sent:.3f}) -> no adj"
            else:
                adj_reason = "no prev_hl or no prediction"

            # Clean up temporary columns if created
            tmp_cols = [c for c in final_df.columns if c.startswith("_tmp_") and c.endswith("_agg")]
            if tmp_cols:
                final_df.drop(columns=tmp_cols, inplace=True, errors="ignore")

        except Exception as e:
            # If anything fails, keep raw prediction
            pred_price = pred_price_raw
            adj_reason = f"sentiment_adj_error: {e}"

        # Now pred_price is adjusted (and pred_price_raw preserved for comparison)

        # --- Load metrics for table ---
        metrics = {
            "lgbm_mae": meta.get("mae_lgbm"),
            "lgbm_r2": meta.get("r2_lgbm"),
            "hybrid_mae": meta.get("mae_hybrid"),
            "hybrid_r2": meta.get("r2_hybrid"),
        }


        # Load top10 textual CSV if present
        csv_path = os.path.join(ART_DIR, f"{ticker}_top10_textual_insights.csv")
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df_top10 = pd.read_csv(csv_path).head(10)
                for _, r in df_top10.iterrows():
                    top10_rows.append({
                        "source": r.get("source", ""),
                        "date": str(r.get("date", "")).split(" ")[0],
                        "pos": f"{float(r.get('pos',0)):.2f}",
                        "neu": f"{float(r.get('neu',0)):.2f}",
                        "neg": f"{float(r.get('neg',0)):.2f}",
                        "text": (r.get("text","")[:140] + "...") if len(str(r.get("text",""))) > 140 else r.get("text","")
                    })
            except Exception:
                pass

        # copy plot if exists
        plot_url = copy_plot_to_static("final_price_prediction_v15.png")

        # --- Generate Gemini summary dynamically for each prediction ---
        gemini_text = None
        try:
            if pipeline and hasattr(pipeline, "generate_gemini_summary"):
                gemini_text = pipeline.generate_gemini_summary(
                    news_docs,
                    reddit_docs,
                    yt_docs,
                    ticker,
                    queries,
                    predicted_price=pred_price,
                    stock_df=stock_df
                )
                # also store it temporarily if you wish
                with open(os.path.join(ART_DIR, f"summary_{ticker}.txt"), "w") as f:
                    f.write(gemini_text or "")
            else:
                gemini_text = "Gemini module not found in pipeline."
        except Exception as e:
            gemini_text = f"Gemini summary failed to generate: {e}"

        # --- Build Event Day → Textual Docs Table (with event type) ---
        event_table = []
        try:
            if "is_event_day" in final_df.columns:
                # Filter only event days (where event flag = 1)
                event_days_df = final_df[final_df["is_event_day"] == 1].copy()

                # Try to get event type column if available, else label as "General Event"
                event_type_col = None
                for c in final_df.columns:
                    if "event_type" in c.lower() or "event_label" in c.lower():
                        event_type_col = c
                        break

                for _, ev_row in event_days_df.iterrows():
                    ev_date = str(ev_row["date"])[:10]
                    ev_type = str(ev_row[event_type_col]) if event_type_col and ev_row[event_type_col] else "General Event"

                    docs_for_day = []
                    for source_name, doc_list in [("News", news_docs), ("Reddit", reddit_docs), ("YouTube", yt_docs)]:
                        for doc in doc_list:
                            # handle Doc class or dict robustly (avoid evaluating doc.get if doc is a Doc)
                            if hasattr(doc, "date"):
                                pub_date = str(doc.date)[:10]
                                title = getattr(doc, "text", "")
                                sentiment_val = getattr(doc, "sentiment", "")
                            elif isinstance(doc, dict):
                                pub_date = str(doc.get("publishedAt", doc.get("date", "")))[:10]
                                title = doc.get("title", "") or doc.get("text", "")
                                sentiment_val = doc.get("sentiment", "")
                            else:
                                pub_date = ""
                                title = ""
                                sentiment_val = ""

                            if ev_date in pub_date:
                                docs_for_day.append({
                                    "date": ev_date,
                                    "event_type": ev_type,
                                    "source": source_name,
                                    "title": (title or "")[:100],
                                    "sentiment": sentiment_val
                                })
                            if len(docs_for_day) >= 3:
                                break


                    if docs_for_day:
                        event_table.extend(docs_for_day)
        except Exception as e:
            print(f"[WARN] Could not create event_table: {e}")
            event_table = []


        # --- Recommendation based on sentiment-adjusted prediction and aggregated sentiment ---
        # Note: this overrides compute_recommendation() result based on adjusted sentiment.
        try:
            # ensure mean_sent, pred_price, pred_price_raw are defined
            if "mean_sent" not in locals():
                mean_sent = 0.0
            # thresholds used earlier for adjustment (keep consistent)
            POS_THRESH = locals().get("POS_THRESH", 0.02)
            NEG_THRESH = locals().get("NEG_THRESH", -0.02)

            # If sentiment is neutral -> HOLD
            if mean_sent is None:
                mean_sent = 0.0

            if NEG_THRESH < mean_sent < POS_THRESH:
                recommendation = "HOLD"
                rec_class = "hold"
                rec_reason = f"Neutral sentiment (mean={mean_sent:.3f})"
            else:
                # If adjusted and raw predictions exist, compare them
                if (pred_price is None) or ('pred_price_raw' not in locals() or pred_price_raw is None):
                    # fallback to original logic if any value missing
                    recommendation, rec_reason, rec_class = compute_recommendation(pred_ret, top10_rows)
                else:
                    if pred_price > pred_price_raw:
                        recommendation = "BUY"
                        rec_class = "buy"
                        rec_reason = f"Adjusted > model: {pred_price_raw:.2f} → {pred_price:.2f} | mean_sent={mean_sent:.3f}"
                    elif pred_price < pred_price_raw:
                        recommendation = "SELL"
                        rec_class = "sell"
                        rec_reason = f"Adjusted < model: {pred_price_raw:.2f} → {pred_price:.2f} | mean_sent={mean_sent:.3f}"
                    else:
                        recommendation = "HOLD"
                        rec_class = "hold"
                        rec_reason = f"No change after sentiment adjustment (mean={mean_sent:.3f})"

        except Exception as e:
            # safe fallback to existing recommendation logic
            recommendation, rec_reason, rec_class = compute_recommendation(pred_ret, top10_rows)

        ts = int(time.time())

        return render_template(
            "result.html",
            ticker=ticker,
            # prev_close=f"{prev_close:.2f}",
            # pred_price=f"{pred_price:.2f}" if pred_price else None,
            prev_close=prev_close,
            pred_price=pred_price, 
            recommendation=recommendation,
            rec_reason=rec_reason,
            rec_class=rec_class,
            metrics=metrics,
            plot_url=plot_url,
            ts=ts,
            top10=top10_rows,
            gemini=gemini_text,
            adj_reason=adj_reason,
            event_table=event_table

        )

    except Exception as e:
        traceback.print_exc()
        flash(f"Unexpected error: {e}", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5691"))
    print(f"Starting Flask (artifacts dir = {ART_DIR}) on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
