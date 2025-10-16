ValidityClassifier — Interactive labeling + CatBoost re-predict demo

Overview
- Flask + HTMX single-page app to iteratively label a pandas DataFrame as valid/invalid.
- Tracks per-row label source: user vs machine.
- Uses CatBoostClassifier to predict labels for non-user rows when you click Re-predict.

Quick start
1) Python 3.10+ recommended.
2) Install dependencies:
   pip install -r requirements.txt
3) Run the app:
   python app.py
4) Open in your browser:
   http://localhost:5000

How it works
- On first load, the app generates a synthetic DataFrame with a key_sale column and some features, plus valid_sale initialized to "unknown".
- Label rows via the inline "valid" / "invalid" buttons. These become canonical (source = user) and will never be re-predicted.
- Once you have at least one valid and one invalid user label, the Re-predict button enables.
- Re-predict trains CatBoost on user-labeled rows only and predicts the rest (source = machine). It also shows a probability percentage next to machine labels.
- You can keep labeling and re-predicting until satisfied.

Notes
- CatBoost must be installed for prediction (see requirements.txt). If unavailable, the app will show an error upon predict.
- Server-side sessions via Flask-Session (filesystem store) keep large data off the cookie.
- Features used for modeling are all columns except key_sale and valid_sale. Categorical columns are passed to CatBoost as categorical.

Reset
- Use the Reset demo data button to regenerate a fresh synthetic dataset.
