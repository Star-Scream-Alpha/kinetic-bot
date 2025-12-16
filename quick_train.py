import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

# Generate synthetic training data that matches your logic
np.random.seed(42)
n_samples = 5000

# Create features
X = pd.DataFrame({
    'kinetic_score': np.random.uniform(10000, 80000, n_samples),
    'vol_pct_5': np.random.uniform(1e6, 5e6, n_samples),
    'vol_pct_25': np.random.uniform(1e6, 5e6, n_samples),
    'vol_pct_50': np.random.uniform(1e6, 5e6, n_samples),
    'vol_pct_75': np.random.uniform(1e6, 5e6, n_samples),
    'vol_pct_95': np.random.uniform(1e6, 5e6, n_samples),
    'price_velocity': np.random.uniform(-5, 5, n_samples),
    'price_acceleration': np.random.uniform(-2, 2, n_samples),
    'volume_mean': np.random.uniform(1e6, 5e6, n_samples),
    'volume_std': np.random.uniform(1e5, 1e6, n_samples),
    'volume_trend': np.random.uniform(-1e5, 1e5, n_samples),
    'hour_sin': np.random.uniform(-1, 1, n_samples),
    'hour_cos': np.random.uniform(-1, 1, n_samples),
    'minute_sin': np.random.uniform(-1, 1, n_samples),
    'hvg_mean_degree': np.random.uniform(1, 8, n_samples),
    'volatility': np.random.uniform(0.1, 0.5, n_samples),
    'atr': np.random.uniform(5, 50, n_samples),
    'price_range': np.random.uniform(10, 100, n_samples),
    'range_ratio': np.random.uniform(0.001, 0.01, n_samples),
    'bb_width': np.random.uniform(0.01, 0.1, n_samples),
    'network_density': np.random.uniform(0.05, 0.2, n_samples),
    'momentum_5': np.random.uniform(-20, 20, n_samples),
    'momentum_10': np.random.uniform(-30, 30, n_samples),
    'momentum_20': np.random.uniform(-50, 50, n_samples),
    'higher_highs': np.random.randint(0, 2, n_samples),
    'lower_lows': np.random.randint(0, 2, n_samples),
    'volume_imbalance': np.random.uniform(-0.5, 0.5, n_samples),
    'vwap_deviation': np.random.uniform(-0.01, 0.01, n_samples),
    'price_trend': np.random.uniform(-0.002, 0.002, n_samples),
    'is_uptrend': np.random.randint(0, 2, n_samples),
})

# Create labels based on YOUR logic
y_m1 = (X['kinetic_score'] > 37500).astype(int)
y_m2 = (X['hvg_mean_degree'] < 3.5).astype(int)
y_m3 = np.where(X['price_trend'] > 0, -1, 1)

# Train Model 1
scaler1 = StandardScaler()
X_scaled1 = scaler1.fit_transform(X)
model1 = xgb.XGBClassifier(n_estimators=100, random_state=42)
model1.fit(X_scaled1, y_m1)

# Train Model 2
scaler2 = StandardScaler()
X_scaled2 = scaler2.fit_transform(X)
model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16), random_state=42, max_iter=100)
model2.fit(X_scaled2, y_m2)

# Train Model 3
scaler3 = StandardScaler()
X_scaled3 = scaler3.fit_transform(X)
model3_clf = RandomForestClassifier(n_estimators=100, random_state=42)
model3_clf.fit(X_scaled3, y_m3)
model3_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model3_reg.fit(X_scaled3, np.abs(y_m3) * 45)  # Expected return

# Save
with open('model1_kinetic_xgboost.pkl', 'wb') as f:
    pickle.dump({'model': model1, 'scaler': scaler1}, f)
with open('model2_regime_nn.pkl', 'wb') as f:
    pickle.dump({'model': model2, 'scaler': scaler2}, f)
with open('model3_direction_rf.pkl', 'wb') as f:
    pickle.dump({'classifier': model3_clf, 'regressor': model3_reg, 'scaler': scaler3}, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("✅ Models trained and saved!")