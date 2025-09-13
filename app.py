import json
import os
from dataclasses import dataclass
from typing import Tuple

from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import pandas as pd

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


def put_frame_in_session(df: pd.DataFrame):
    session[SessionKeys.df_json] = df.to_json(orient="split")


def get_frame_from_session() -> pd.DataFrame:
    df_json = session.get(SessionKeys.df_json)
    if df_json is None:
        df = make_demo_dataframe()
        put_frame_in_session(df)
        # initialize label sources as unknown
        session[SessionKeys.label_source] = {k: "unknown" for k in df["key_primary"].tolist()}
        session[SessionKeys.proba] = {}
        return df
    df = pd.read_json(df_json, orient="split")
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


@app.route("/")
def index():
    df = get_frame_from_session()
    src = get_sources()
    pos, neg = labeled_counts()
    can_predict = pos > 0 and neg > 0 and CatBoostClassifier is not None
    return render_template(
        "index.html",
        df=df,
        sources=src,
        can_predict=can_predict,
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
    if key not in set(df["key_primary"].astype(str)):
        return ("Unknown key", 404)

    # Update the row
    df.loc[df["key_primary"] == key, "valid_sale"] = value
    put_frame_in_session(df)

    # Mark as user-labeled
    set_source(key, "user")
    # Clear any previous proba for this key (since it's canonical now)
    probas = get_probas()
    if key in probas:
        probas.pop(key, None)
        session[SessionKeys.proba] = probas

    # Return updated table body fragment
    src = get_sources()
    pos, neg = labeled_counts()
    can_predict = pos > 0 and neg > 0 and CatBoostClassifier is not None
    return render_template("_table.html", df=df, sources=src, probas=get_probas(), can_predict=can_predict)


@app.post("/predict")
def predict():
    df = get_frame_from_session()
    src = get_sources()

    if CatBoostClassifier is None:
        return ("CatBoost is not installed in this environment.", 500)

    # Prepare training data: only user-labeled rows
    user_mask = df["key_primary"].map(src).eq("user")
    train_df = df[user_mask]

    # Guard: need at least one valid and one invalid
    y_train = train_df["valid_sale"].map({"valid": 1, "invalid": 0})
    if y_train.dropna().nunique() < 2:
        return ("Need at least one 'valid' and one 'invalid' user label before predicting.", 400)

    # Feature set: all columns except key_primary and valid_sale
    feature_cols = [c for c in df.columns if c not in ("key_primary", "valid_sale")]
    X_train = train_df[feature_cols]

    # Identify categorical columns by dtype object
    cat_cols = [i for i, c in enumerate(feature_cols) if X_train[c].dtype == "object"]

    # Build and fit CatBoost
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        iterations=200,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    # Convert y to numeric
    y = y_train.astype("Int64").astype(int)

    model.fit(X_train, y, cat_features=cat_cols)

    # Predict for non-user rows (unknown or machine)
    non_user_mask = ~user_mask
    X_pred = df.loc[non_user_mask, feature_cols]
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
    can_predict = pos > 0 and neg > 0 and CatBoostClassifier is not None
    return render_template("_table.html", df=df, sources=get_sources(), probas=get_probas(), can_predict=can_predict)


@app.post("/reset")
def reset():
    df = make_demo_dataframe()
    put_frame_in_session(df)
    session[SessionKeys.label_source] = {k: "unknown" for k in df["key_primary"].tolist()}
    session[SessionKeys.proba] = {}
    return redirect(url_for("index"))


# Templates expect Jinja2 available. Provide a simple run entry.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
