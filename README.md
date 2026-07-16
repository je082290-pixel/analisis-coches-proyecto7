Nasdaq100-Prediction
A research notebook that builds and backtests a 1-month (H=20 trading days) prediction and execution pipeline for QQQ (Nasdaq-100 ETF). It uses an ensemble (XGBoost + Ridge stacking), isotonic calibration, conditional uncertainty estimation, a logistic sign-gate, regime-aware feature selection (VIX terciles), walk‑forward backtesting and sizing based on a Kelly-style conviction filter. Intended for research and experimentation — not financial advice.

Key features
Walk‑forward evaluation (rolling folds) with train / validation / test splits.
Ensemble modeling: XGBRegressor + RidgeCV with QuantileTransformer, stacked by a linear meta-model.
Out-of-fold isotonic calibration and sigma (uncertainty) estimation from residuals.
Sign gate (LogisticRegression) to filter low‑confidence trades.
Regimes by volatility (VIX z-score terciles) and Top‑K feature selection per regime.
Sizing/execution: Kelly-derived weights, vol‑targeting, edge = mu - λ·σ, transaction cost stress tests.
Saves per-fold signals/weights/metrics as CSV for downstream analysis.
Repository contents
Nasdaq_100_(1).ipynb — single Jupyter notebook containing the full data download, feature engineering, modeling, walk‑forward loop, backtest and reporting. Outputs CSVs to signals_out/.
Requirements
The notebook imports the following packages (use these versions as a starting point):

Python 3.8+
numpy
pandas
yfinance
scikit-learn
xgboost
Suggested pip install command:

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install numpy pandas yfinance scikit-learn xgboost
