# Nasdaq100 Prediction
Advanced machine learning framework for predicting QQQ (Nasdaq100) price movements 20 days ahead using ensemble methods, volatility regimes, and quantile calibration.

What This Is
This project implements a sophisticated quantitative trading strategy for QQQ that:

Predicts 20-day forward returns using an ensemble of XGBoost and Ridge regression models with isotonic calibration
Manages uncertainty through conditional volatility estimation and Kelly criterion sizing
Adapts to market regimes by training separate models under different VIX volatility levels (low/medium/high)
Optimizes execution with walk-forward backtesting, position sizing with edge-aware cost filters, and transaction cost stress testing
The notebook runs a complete walk-forward validation across 13 folds from 1999–2025, achieving ~14% average CAGR with 0.81 Sharpe ratio.

Stack
Language: Python 3
Notebook environment: Jupyter/Google Colab
ML framework: XGBoost, scikit-learn (ensemble, preprocessing, calibration)
Data: yfinance (QQQ + 9 external factors: VIX, VXN, bonds, commodities, credit)
Key dependencies:
xgboost – gradient boosting for feature importance and predictions
scikit-learn – RidgeCV, IsotonicRegression, LogisticRegression, QuantileTransformer
pandas, numpy – data manipulation and math
yfinance – market data fetching

**Author:** Juan David España Hinestrosa
