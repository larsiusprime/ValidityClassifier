import json
import os
from dataclasses import dataclass
from typing import Tuple

from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

# Try to import CatBoost; provide a gentle error if unavailable
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


@dataclass
class SessionKeys:
    df_json: str = "df_json"
    label_source: str = "label_source"  # map from key_primary to source: user|machine|unknown
    proba: str = "pred_proba"  # optional prediction probability per key


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# Configure server-side sessions via Flask-Session (filesystem store)
session_dir = os.path.join(os.path.dirname(__file__), ".flask_session")
os.makedirs(session_dir, exist_ok=True)
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=session_dir,
    SESSION_PERMANENT=False,
)
Session(app)


def make_demo_dataframe(n: int = 200) -> pd.DataFrame:
    """Create a simple synthetic dataset with a key_primary column and mixed features."""
    import numpy as np

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "key_primary": [f"row_{i:04d}" for i in range(n)],
        "amount": rng.normal(100, 30, n).round(2),
        "items": rng.integers(1, 10, n),
        "channel": rng.choice(["online", "store", "phone"], n, p=[0.5, 0.4, 0.1]),
        "region": rng.choice(["NA", "EU", "APAC"], n),
        # a weak signal for validity to make the demo feel responsive
        "discount": rng.uniform(0, 0.5, n).round(3),
    })
    # Initialize target column
    df["valid_sale"] = "unknown"
    return df


def load_initial_dataframe() -> pd.DataFrame:
    """Load initial DataFrame from sales_data.csv if present, else use demo data.

    - Looks for sales_data.csv in the project directory (same dir as this app.py).
    - Ensures a 'valid_sale' column exists and is initialized to 'unknown'.
    - Ensures key_primary is treated as string.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)
            # Enforce key as string to be consistent throughout the app
            if "key_primary" in df.columns:
                df["key_primary"] = df["key_primary"].astype(str)
            else:
                # If CSV somehow lacks key_primary, fall back to demo to avoid breakage
                return make_demo_dataframe()
            if "valid_sale" not in df.columns:
                df["valid_sale"] = "unknown"
            return df
        except Exception:
            # On any read/parse error, fall back to demo data
            print("Unable to find sales_data.csv. Using synthetic dataframe.")
            return make_demo_dataframe()
    # Default: no CSV found
    return make_demo_dataframe()


def sanitize_df(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Sanitize feature columns once per session to stabilize dtypes.

    - Object-like columns => convert to Python str with empty string for missing.
    - Numeric-like columns => coerce to numeric, fill missing with 0.
    """
    df = df.copy()
    for c in feature_cols:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype("string").fillna("").astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def put_frame_in_session(df: pd.DataFrame):
    session[SessionKeys.df_json] = df.to_json(orient="split")


def get_frame_from_session() -> pd.DataFrame:
    df_json = session.get(SessionKeys.df_json)
    if df_json is None:
        df = load_initial_dataframe()
        # Sanitize once per session for feature columns
        feature_cols = [c for c in df.columns if c not in ("key_primary", "valid_sale")]
        df = sanitize_df(df, feature_cols)
        put_frame_in_session(df)
        # initialize label sources as unknown
        session[SessionKeys.label_source] = {k: "unknown" for k in df["key_primary"].tolist()}
        session[SessionKeys.proba] = {}
        return df
    df = pd.read_json(StringIO(df_json), orient="split")
    # Ensure key_primary is string after JSON round-trip to avoid dtype drift
    if "key_primary" in df.columns:
        df["key_primary"] = df["key_primary"].astype(str)
    return df


def get_sources() -> dict:
    # ensure consistency with current df keys (handle first run)
    df = get_frame_from_session()
    src = session.get(SessionKeys.label_source)
    if not src:
        src = {k: "unknown" for k in df["key_primary"].tolist()}
        session[SessionKeys.label_source] = src
    return src


def set_source(key: str, source: str):
    src = get_sources()
    src[key] = source
    session[SessionKeys.label_source] = src


def get_probas() -> dict:
    return session.get(SessionKeys.proba) or {}


def set_proba(key: str, proba: float):
    probas = get_probas()
    probas[key] = float(proba)
    session[SessionKeys.proba] = probas


def labeled_counts() -> Tuple[int, int]:
    """Return counts of user-labeled valid and invalid rows."""
    df = get_frame_from_session()
    src = get_sources()
    user_mask = df["key_primary"].map(src).eq("user")
    user_df = df[user_mask]
    pos = (user_df["valid_sale"] == "valid").sum()
    neg = (user_df["valid_sale"] == "invalid").sum()
    return int(pos), int(neg)


# Minimum per-class labels required to enable prediction with stratified split
REQUIRED_PER_CLASS = 3


def predict_button_state(pos: int, neg: int) -> dict:
    """Compute enablement and UX strings for the predict button.

    Returns dict with keys: can_predict (bool), predict_text (str), predict_title (str)
    """
    can_predict = (
        CatBoostClassifier is not None and pos >= REQUIRED_PER_CLASS and neg >= REQUIRED_PER_CLASS
    )
    if can_predict:
        text = "Re-predict"
        title = "Run the model on unlabeled rows"
    else:
        need_pos = max(0, REQUIRED_PER_CLASS - pos)
        need_neg = max(0, REQUIRED_PER_CLASS - neg)
        parts = []
        if need_pos:
            parts.append(f"{need_pos} more valid")
        if need_neg:
            parts.append(f"{need_neg} more invalid")
        needed = " and ".join(parts) if parts else f"{REQUIRED_PER_CLASS} per class"
        text = f"Re-predict (need {REQUIRED_PER_CLASS}/class)"
        title = f"Label {needed} to enable predictions"
    return {"can_predict": can_predict, "predict_text": text, "predict_title": title}


@app.route("/")
def index():
    df = get_frame_from_session()
    src = get_sources()
    pos, neg = labeled_counts()
    state = predict_button_state(pos, neg)
    return render_template(
        "index.html",
        df=df,
        sources=src,
        can_predict=state["can_predict"],
        predict_text=state["predict_text"],
        predict_title=state["predict_title"],
        pos=pos,
        neg=neg,
        probas=get_probas(),
    )


@app.post("/label")
def set_label():
    """Set a user label for a specific row identified by key_primary.

    Expects form fields: key, value in {valid, invalid}
    Returns the updated table fragment.
    """
    key = request.form.get("key")
    value = request.form.get("value")
    if key is None or value not in {"valid", "invalid"}:
        return ("Invalid parameters", 400)

    df = get_frame_from_session()
    key = str(key)
    keys_str = set(df["key_primary"].astype(str))
    if key not in keys_str:
        return ("Unknown key", 404)

    # Update the row using string-safe comparison
    mask = df["key_primary"].astype(str) == key
    df.loc[mask, "valid_sale"] = value
    put_frame_in_session(df)

    # Mark as user-labeled
    set_source(key, "user")
    # Clear any previous proba for this key (since it's canonical now)
    probas = get_probas()
    if key in probas:
        probas.pop(key, None)
        session[SessionKeys.proba] = probas

    # Return updated single row fragment + OOB updates for status/predict
    src = get_sources()
    pos, neg = labeled_counts()
    state = predict_button_state(pos, neg)

    # get the updated row as a Series
    row = df.loc[mask].iloc[0]

    return render_template(
        "_row_response.html",
        df=df,
        row=row,
        sources=src,
        probas=get_probas(),
        pos=pos,
        neg=neg,
        can_predict=state["can_predict"],
        predict_text=state["predict_text"],
        predict_title=state["predict_title"],
    )


@app.post("/unset")
def unset_label():
    """Unset a user label for a specific row (revert to unknown)."""
    key = request.form.get("key")
    if key is None:
        return ("Invalid parameters", 400)

    df = get_frame_from_session()
    key = str(key)
    keys_str = set(df["key_primary"].astype(str))
    if key not in keys_str:
        return ("Unknown key", 404)

    # Revert the row label to unknown using string-safe comparison
    mask = df["key_primary"].astype(str) == key
    df.loc[mask, "valid_sale"] = "unknown"
    put_frame_in_session(df)

    # Mark source as unknown again
    set_source(key, "unknown")

    # Clear any stored probability for this key
    probas = get_probas()
    if key in probas:
        probas.pop(key, None)
        session[SessionKeys.proba] = probas

    # Prepare response similar to /label
    src = get_sources()
    pos, neg = labeled_counts()
    state = predict_button_state(pos, neg)

    row = df.loc[df["key_primary"] == key].iloc[0]

    return render_template(
        "_row_response.html",
        df=df,
        row=row,
        sources=src,
        probas=get_probas(),
        pos=pos,
        neg=neg,
        can_predict=state["can_predict"],
        predict_text=state["predict_text"],
        predict_title=state["predict_title"],
    )


@app.post("/predict")
def predict():
    df = get_frame_from_session()
    src = get_sources()

    if CatBoostClassifier is None:
        return ("CatBoost is not installed in this environment.", 500)

    # Prepare training data: only user-labeled rows
    user_mask = df["key_primary"].map(src).eq("user")
    train_df = df[user_mask]

    # Guard: need at least REQUIRED_PER_CLASS labels per class for stratified split
    y_train = train_df["valid_sale"].map({"valid": 1, "invalid": 0})
    counts = y_train.value_counts()
    if counts.get(1, 0) < REQUIRED_PER_CLASS or counts.get(0, 0) < REQUIRED_PER_CLASS:
        return (
            f"Need at least {REQUIRED_PER_CLASS} 'valid' and {REQUIRED_PER_CLASS} 'invalid' user labels before predicting.",
            400,
        )

    # Feature set: all columns except key_primary and valid_sale
    feature_cols = [c for c in df.columns if c not in ("key_primary", "valid_sale")]
    X_train = train_df[feature_cols]

    # Identify categorical columns by dtype object
    cat_cols = [i for i, c in enumerate(feature_cols) if X_train[c].dtype == "object"]

    # Build and fit CatBoost with early stopping and all CPU cores
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        iterations=200,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    # Convert y to numeric
    y = y_train.astype("Int64").astype(int)

    # Train/validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, stratify=y, random_state=42)

    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )

    # Predict for non-user rows (unknown or machine)
    non_user_mask = ~user_mask
    X_pred = df.loc[non_user_mask, feature_cols].copy()
    if not X_pred.empty:
        proba = model.predict_proba(X_pred)[:, 1]
        pred_class = (proba >= 0.5).astype(int)
        pred_label = pd.Series(pred_class, index=X_pred.index).map({1: "valid", 0: "invalid"})

        df.loc[X_pred.index, "valid_sale"] = pred_label
        put_frame_in_session(df)

        # Update sources and store probabilities
        for idx, p in zip(df.loc[X_pred.index, "key_primary"], proba):
            set_source(str(idx), "machine")
            set_proba(str(idx), float(p))

    pos, neg = labeled_counts()
    state = predict_button_state(pos, neg)
    return render_template(
        "_table.html",
        df=df,
        sources=get_sources(),
        probas=get_probas(),
        can_predict=state["can_predict"],
        predict_text=state["predict_text"],
        predict_title=state["predict_title"],
        pos=pos,
        neg=neg,
    )


@app.post("/reset")
def reset():
    df = load_initial_dataframe()
    # Sanitize once per session on reset
    feature_cols = [c for c in df.columns if c not in ("key_primary", "valid_sale")]
    df = sanitize_df(df, feature_cols)
    put_frame_in_session(df)
    session[SessionKeys.label_source] = {k: "unknown" for k in df["key_primary"].tolist()}
    session[SessionKeys.proba] = {}
    return redirect(url_for("index"))


# Templates expect Jinja2 available. Provide a simple run entry.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
