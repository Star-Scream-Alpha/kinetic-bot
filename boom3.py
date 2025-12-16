"""
Kinetic Hunter Pro - ML-Powered Trading Dashboard
=================================================
Uses three trained ML models for algorithmic trading
"""

import boto3
import pandas as pd
import numpy as np
import datetime
import time
from collections import deque
from io import BytesIO
from numba import njit
import streamlit as st
import pickle

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Kinetic Hunter Pro ML", layout="wide", page_icon="🤖")

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
        color: #8b9dc3 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .status-scanning {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
    }
    
    .status-cooldown {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .ml-score {
        background: rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

BUCKET = "live-market-data"
SYMBOL = "NIFTY"

START_DATE = datetime.date(2025, 9, 1)
END_DATE = datetime.date(2025, 11, 26)
DEFAULT_DATE = datetime.date(2025, 11, 25)

MARKET_START = datetime.time(9, 15)
MARKET_END = datetime.time(15, 30)

EXPIRY_SEP = datetime.date(2025, 9, 25)
EXPIRY_OCT = datetime.date(2025, 10, 30)
EXPIRY_NOV = datetime.date(2025, 11, 27)

HVG_WINDOW = 50

# ==========================================
# LOAD ML MODELS
# ==========================================
@st.cache_resource
def load_ml_models():
    """Load trained ML models"""
    try:
        with open('model1_kinetic_xgboost.pkl', 'rb') as f:
            model1_data = pickle.load(f)
        
        with open('model2_regime_nn.pkl', 'rb') as f:
            model2_data = pickle.load(f)
        
        with open('model3_direction_rf.pkl', 'rb') as f:
            model3_data = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return model1_data, model2_data, model3_data, feature_names
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run train_models.py first.")
        return None, None, None, None

# ==========================================
# HVG CALCULATION
# ==========================================
@njit
def calculate_hvg_mean_degree(prices):
    n = len(prices)
    if n < 2: return 0.0
    
    total_degree = 0
    for i in range(n):
        current_degree = 0
        if i > 0:
            max_h = -99999999.0
            for j in range(i - 1, -1, -1):
                if j == i - 1: 
                    current_degree += 1; max_h = prices[j]
                elif prices[j] > max_h:
                    current_degree += 1; max_h = prices[j]
                if prices[j] >= prices[i]: break
        if i < n - 1:
            max_h = -99999999.0
            for j in range(i + 1, n):
                if j == i + 1:
                    current_degree += 1; max_h = prices[j]
                elif prices[j] > max_h:
                    current_degree += 1; max_h = prices[j]
                if prices[j] >= prices[i]: break
        total_degree += current_degree
    return total_degree / n

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(window_data):
    """Extract features from window (same as training)"""
    
    prices = window_data['LTP'].values
    volumes = window_data['Volume'].values
    
    features = {}
    
    # Kinetic Score
    vol_diff = np.diff(volumes)
    trade_vol = np.where(vol_diff > 0, vol_diff, 0)
    displacement = abs(prices[-1] - prices[0]) + 0.05
    features['kinetic_score'] = np.sum(trade_vol) / displacement
    
    # Volume Percentiles
    features['vol_pct_5'] = np.percentile(volumes, 5)
    features['vol_pct_25'] = np.percentile(volumes, 25)
    features['vol_pct_50'] = np.percentile(volumes, 50)
    features['vol_pct_75'] = np.percentile(volumes, 75)
    features['vol_pct_95'] = np.percentile(volumes, 95)
    
    # Price Velocity & Acceleration
    price_changes = np.diff(prices)
    features['price_velocity'] = np.mean(price_changes)
    features['price_acceleration'] = np.mean(np.diff(price_changes)) if len(price_changes) > 1 else 0
    
    # Volume Stats
    features['volume_mean'] = np.mean(volumes)
    features['volume_std'] = np.std(volumes)
    features['volume_trend'] = volumes[-1] - volumes[0]
    
    # Time Features
    hour = window_data['DateTime'].iloc[-1].hour
    minute = window_data['DateTime'].iloc[-1].minute
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['minute_sin'] = np.sin(2 * np.pi * minute / 60)
    
    # HVG Features
    hvg_degree = calculate_hvg_mean_degree(prices)
    features['hvg_mean_degree'] = hvg_degree
    
    # Volatility
    returns = price_changes / prices[:-1]
    features['volatility'] = np.std(returns) * np.sqrt(252 * 6.5 * 60)
    features['atr'] = np.mean(np.abs(price_changes[-20:])) if len(price_changes) >= 20 else np.mean(np.abs(price_changes))
    
    # Price Range
    features['price_range'] = prices.max() - prices.min()
    features['range_ratio'] = features['price_range'] / prices[-1]
    
    # Bollinger Band Width
    sma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    std = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
    features['bb_width'] = (2 * std) / sma if sma > 0 else 0
    
    # Network Density
    features['network_density'] = hvg_degree / len(prices)
    
    # Momentum Features
    features['momentum_5'] = prices[-1] - prices[-5] if len(prices) >= 5 else 0
    features['momentum_10'] = prices[-1] - prices[-10] if len(prices) >= 10 else 0
    features['momentum_20'] = prices[-1] - prices[-20] if len(prices) >= 20 else 0
    
    # Price Pattern
    if len(prices) >= 10:
        features['higher_highs'] = int(prices[-5:].max() > prices[-10:-5].max())
        features['lower_lows'] = int(prices[-5:].min() < prices[-10:-5].min())
    else:
        features['higher_highs'] = 0
        features['lower_lows'] = 0
    
    # Volume Imbalance
    recent_vol = volumes[-10:].sum() if len(volumes) >= 10 else volumes.sum()
    older_vol = volumes[-20:-10].sum() if len(volumes) >= 20 else recent_vol
    features['volume_imbalance'] = (recent_vol - older_vol) / (older_vol + 1)
    
    # VWAP
    vwap = np.sum(prices * volumes) / np.sum(volumes)
    features['vwap_deviation'] = (prices[-1] - vwap) / vwap
    
    # Trend
    features['price_trend'] = (prices[-1] - prices[0]) / prices[0]
    features['is_uptrend'] = int(features['price_trend'] > 0)
    
    return features

# ==========================================
# ML BRAIN
# ==========================================
class MLTradingBrain:
    def __init__(self, model1_data, model2_data, model3_data, feature_names):
        self.model1 = model1_data['model']
        self.scaler1 = model1_data['scaler']
        
        self.model2 = model2_data['model']
        self.scaler2 = model2_data['scaler']
        
        self.model3_clf = model3_data['classifier']
        self.model3_reg = model3_data['regressor']
        self.scaler3 = model3_data['scaler']
        
        self.feature_names = feature_names
        
        self.last_m1_score = 0
        self.last_m2_class = 0
        self.last_m2_confidence = 0
        self.last_m3_direction = 0
        self.last_m3_return = 0
    
    def process_window(self, window_data):
        """Run all three models on a window of data"""
        
        # Extract features
        features_dict = extract_features(window_data)
        features_df = pd.DataFrame([features_dict])
        
        # Ensure feature order matches training
        features_df = features_df[self.feature_names]
        
        # MODEL 1: Kinetic Energy Classifier
        X1 = self.scaler1.transform(features_df)
        m1_proba = self.model1.predict_proba(X1)[0, 1]
        self.last_m1_score = m1_proba
        
        if m1_proba < 0.75:
            return 0  # Filter 1 failed
        
        # MODEL 2: Regime Classifier
        X2 = self.scaler2.transform(features_df)
        m2_class = self.model2.predict(X2)[0]
        m2_probas = self.model2.predict_proba(X2)[0]
        self.last_m2_class = m2_class
        self.last_m2_confidence = m2_probas[m2_class]
        
        if m2_class != 1:  # Not Pre-Breakout
            return 0  # Filter 2 failed
        
        # MODEL 3: Direction Predictor
        X3 = self.scaler3.transform(features_df)
        m3_direction = self.model3_clf.predict(X3)[0]
        m3_proba = self.model3_clf.predict_proba(X3)[0]
        m3_return = self.model3_reg.predict(X3)[0]
        
        self.last_m3_direction = m3_direction
        self.last_m3_return = m3_return
        
        # Final decision logic
        direction_confidence = max(m3_proba)
        
        if direction_confidence > 0.65 and m3_return > 30:
            return m3_direction  # 1 for Long, -1 for Short
        
        return 0  # No trade

# ==========================================
# DATA LOADING
# ==========================================
def get_trading_symbol(current_date):
    if current_date <= EXPIRY_SEP: return "NIFTY25SEPFUT"
    elif current_date <= EXPIRY_OCT: return "NIFTY25OCTFUT"
    else: return "NIFTY25NOVFUT"

@st.cache_data(ttl=3600)
def get_data_for_date(date_obj):
    s3 = boto3.client("s3")
    year = date_obj.year; month = date_obj.month; day = date_obj.day
    ts = get_trading_symbol(date_obj)
    key = f"year={year}/month={month:02d}/day={day:02d}/Futures/{SYMBOL}/{ts}.parquet"
    
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        
        if 'DateTime' not in df.columns:
            df['DateTime'] = pd.to_datetime(
                df['Date'].astype(str) + " " + df['Time'].astype(str), 
                dayfirst=True, errors='coerce'
            )

        col_map = {'LastTradedPrice': 'LTP', 'Close': 'LTP'}
        df.rename(columns=col_map, inplace=True)
        
        if 'Volume' not in df.columns:
            if 'OpenInterest' in df.columns: df['Volume'] = df['OpenInterest']
            elif 'LTQ' in df.columns: df['Volume'] = df['LTQ'] 
            else: df['Volume'] = 0
            
        df = df.dropna(subset=['DateTime', 'LTP']).sort_values('DateTime').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ==========================================
# SIMULATION ENGINE
# ==========================================
def run_simulation(df, params, ml_brain):
    state = 'SEARCHING'
    
    entry_price = 0.0
    entry_time = None
    direction = 0 
    cooldown_start = None
    
    daily_pnl = 0.0
    trades_log = []
    
    tick_count = 0
    
    yield {
        "ts": df.iloc[0]['DateTime'], "ltp": df.iloc[0]['LTP'], 
        "m1_score": 0, "m2_class": 0, "m2_conf": 0, 
        "m3_dir": 0, "m3_ret": 0,
        "state": state, "pnl": 0, "total_pnl": 0, "dir": 0, "log": [], "entry_price": 0
    }

    for i in range(HVG_WINDOW, len(df)):
        row = df.iloc[i]
        ts = row['DateTime']
        ltp = row['LTP']
        curr_time = ts.time()
        
        tick_count += 1
        exit_reason = None
        
        # COOLDOWN STATE
        if state == 'COOLDOWN':
            if (ts - cooldown_start).total_seconds() > params['cooldown_seconds']:
                state = 'SEARCHING'
        
        # SEARCHING STATE
        elif state == 'SEARCHING':
            window = df.iloc[i-HVG_WINDOW:i]
            sig = ml_brain.process_window(window)
            
            is_market_open = (curr_time >= MARKET_START) and (curr_time < MARKET_END)
            
            if sig != 0 and is_market_open:
                state = 'IN_TRADE'
                direction = sig
                entry_price = ltp
                entry_time = ts
                
        # IN TRADE STATE
        elif state == 'IN_TRADE':
            elapsed = (ts - entry_time).total_seconds()
            
            if direction == 1: pnl_pts = ltp - entry_price
            else: pnl_pts = entry_price - ltp
            
            # Exits
            if pnl_pts <= -params['stop_loss_points']: exit_reason = "SL"
            elif pnl_pts >= params['take_profit_points']: exit_reason = "TP"
            elif elapsed >= params['max_hold_seconds']: exit_reason = "TIME"
            elif curr_time >= MARKET_END: exit_reason = "EOD"
            
            if exit_reason:
                net_pts = pnl_pts - params['cost_per_trade']
                realized_pnl = net_pts * params['qty']
                daily_pnl += realized_pnl
                
                trades_log.insert(0, {
                    "Entry Time": entry_time.strftime("%H:%M:%S"),
                    "Exit Time": ts.strftime("%H:%M:%S"),
                    "Type": "LONG" if direction == 1 else "SHORT",
                    "Entry": f"{entry_price:.2f}",
                    "Exit": f"{ltp:.2f}",
                    "PnL": realized_pnl,
                    "Pts": net_pts,
                    "Reason": exit_reason
                })
                
                state = 'COOLDOWN'
                cooldown_start = ts
                direction = 0
                entry_price = 0.0

        # Floating PnL
        float_pnl = 0.0
        if state == 'IN_TRADE':
            if direction == 1: float_pnl = (ltp - entry_price) * params['qty']
            else: float_pnl = (entry_price - ltp) * params['qty']

        # Yield Update
        if tick_count % 10 == 0 or state == 'IN_TRADE' or exit_reason or state == 'COOLDOWN':
            yield {
                "ts": ts, "ltp": ltp,
                "m1_score": ml_brain.last_m1_score,
                "m2_class": ml_brain.last_m2_class,
                "m2_conf": ml_brain.last_m2_confidence,
                "m3_dir": ml_brain.last_m3_direction,
                "m3_ret": ml_brain.last_m3_return,
                "state": state, "pnl": float_pnl,
                "total_pnl": daily_pnl, "dir": direction,
                "log": trades_log, "entry_price": entry_price
            }

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.title("🤖 Kinetic Hunter Pro ML")
    st.markdown("### *Three ML Models Working Together*")
    st.markdown("---")
    
    # Load ML Models
    model1_data, model2_data, model3_data, feature_names = load_ml_models()
    
    if model1_data is None:
        st.stop()
    
    st.success("✅ ML Models Loaded Successfully")
    
    # Sidebar
    st.sidebar.header("🎯 Simulation Control")
    st.sidebar.markdown("---")
    
    selected_date = st.sidebar.date_input(
        "📅 Select Date", DEFAULT_DATE, min_value=START_DATE, max_value=END_DATE
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Replay Speed")
    
    speed = st.sidebar.select_slider(
        "⚡ Speed", 
        options=["Realtime", "Fast", "Hyper", "Warp", "Instant"], 
        value="Fast"
    )
    speed_map = {"Realtime": 0.1, "Fast": 0.01, "Hyper": 0.001, "Warp": 0.0001, "Instant": 0}
    sleep_time = speed_map[speed]
    
    time_compression = st.sidebar.checkbox("⏰ Time Compression (1 min = 1 sec)", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 Risk Management")
    
    stop_loss = st.sidebar.number_input("Stop Loss (pts)", 5.0, 100.0, 20.0, 5.0)
    take_profit = st.sidebar.number_input("Take Profit (pts)", 10.0, 200.0, 60.0, 5.0)
    max_hold = st.sidebar.number_input("Max Hold (min)", 1, 120, 30, 5)
    cooldown = st.sidebar.number_input("Cooldown (min)", 1, 30, 5, 1)
    qty = st.sidebar.number_input("Quantity", 1, 10000, 450, 25)
    
    params = {
        'stop_loss_points': stop_loss,
        'take_profit_points': take_profit,
        'max_hold_seconds': max_hold * 60,
        'cooldown_seconds': cooldown * 60,
        'qty': qty,
        'cost_per_trade': 1.0
    }
    
    if st.sidebar.button("🚀 Launch", type="primary", use_container_width=True):
        run_dashboard(selected_date, sleep_time, time_compression, params, 
                     model1_data, model2_data, model3_data, feature_names)

def run_dashboard(date_obj, sleep_time, time_compression, params, 
                  model1_data, model2_data, model3_data, feature_names):
    
    with st.spinner(f"⏳ Loading data..."):
        df = get_data_for_date(date_obj)
    
    if df is None or df.empty:
        st.error("❌ No data found")
        return
    
    st.success(f"✅ Loaded {len(df):,} ticks")
    
    # Initialize ML Brain
    ml_brain = MLTradingBrain(model1_data, model2_data, model3_data, feature_names)
    
    # Layout
    st.markdown("### 📈 Live Dashboard")
    kpi_cols = st.columns(5)
    time_metric = kpi_cols[0].empty()
    price_metric = kpi_cols[1].empty()
    pnl_metric = kpi_cols[2].empty()
    status_metric = kpi_cols[3].empty()
    entry_metric = kpi_cols[4].empty()
    
    status_banner = st.empty()
    st.markdown("---")
    
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        st.markdown("##### 🤖 ML MODEL SCORES")
        ml_panel = st.empty()
        
    with col_right:
        st.markdown("##### 📜 TRADE JOURNAL")
        log_placeholder = st.empty()
        st.markdown("---")
        st.markdown("##### 📊 SESSION STATS")
        stats_placeholder = st.empty()
    
    # Run Simulation
    sim = run_simulation(df, params, ml_brain)
    last_tick_timestamp = None
    
    for tick in sim:
        # Update Metrics
        time_metric.metric("⏰ TIME", tick['ts'].strftime("%H:%M:%S"))
        price_metric.metric("💹 PRICE", f"₹{tick['ltp']:.2f}")
        
        total_val = tick['total_pnl'] + tick['pnl']
        pnl_metric.metric("💰 P&L", f"₹{total_val:,.0f}", 
                         delta=f"₹{tick['pnl']:.0f}" if tick['pnl'] != 0 else None)
        
        if tick['entry_price'] > 0:
            entry_metric.metric("🎯 ENTRY", f"₹{tick['entry_price']:.2f}")
        else:
            entry_metric.metric("🎯 ENTRY", "-")
        
        # Status Banner
        s = tick['state']
        if s == 'SEARCHING':
            status_metric.metric("STATUS", "SCANNING")
            status_banner.markdown('<div class="status-scanning">📡 ML MODELS SCANNING...</div>', 
                                  unsafe_allow_html=True)
        elif s == 'IN_TRADE':
            d_str = "LONG 🟢" if tick['dir'] == 1 else "SHORT 🔴"
            status_metric.metric("STATUS", "ACTIVE")
            status_banner.markdown(f'<div class="status-active">⚡ {d_str} | Float: ₹{tick["pnl"]:.0f}</div>', 
                                  unsafe_allow_html=True)
        elif s == 'COOLDOWN':
            status_metric.metric("STATUS", "COOLDOWN")
            status_banner.markdown('<div class="status-cooldown">❄️ COOLDOWN MODE</div>', 
                                  unsafe_allow_html=True)
        
        # ML Scores Panel
        regime_names = ['Consolidation', 'Pre-Breakout', 'Trending']
        direction_map = {-1: 'SHORT', 0: 'NONE', 1: 'LONG'}
        
        ml_panel.markdown(f"""
        <div class="ml-score">
            <strong>Model 1 (XGBoost):</strong> Kinetic Score = {tick['m1_score']:.3f}
            <br>Status: {'✅ PASS' if tick['m1_score'] > 0.75 else '❌ FAIL'}
        </div>
        <div class="ml-score">
            <strong>Model 2 (Neural Net):</strong> Regime = {regime_names[int(tick['m2_class'])]} ({tick['m2_conf']:.2f})
            <br>Status: {'✅ PASS' if tick['m2_class'] == 1 else '❌ FAIL'}
        </div>
        <div class="ml-score">
            <strong>Model 3 (Random Forest):</strong> Direction = {direction_map.get(int(tick['m3_dir']), 'NONE')}
            <br>Expected Return: {tick['m3_ret']:.1f} points
        </div>
        """, unsafe_allow_html=True)
        
        # Trade Log
        if tick['log']:
            df_log = pd.DataFrame(tick['log'])
            df_log['PnL Display'] = df_log['PnL'].apply(lambda x: f"₹{x:,.0f}")
            df_log['Pts Display'] = df_log['Pts'].apply(lambda x: f"{x:.1f}")
            
            df_display = df_log[['Entry Time', 'Exit Time', 'Type', 'Entry', 'Exit']].copy()
            df_display['PnL'] = df_log['PnL Display']
            df_display['Pts'] = df_log['Pts Display']
            df_display['Reason'] = df_log['Reason']
            
            def style_row(row):
                pnl_val = df_log.loc[row.name, 'PnL']
                if pnl_val > 0:
                    return ['background-color: rgba(17, 153, 142, 0.2); color: #38ef7d'] * len(row)
                else:
                    return ['background-color: rgba(245, 87, 108, 0.2); color: #f5576c'] * len(row)
            
            log_placeholder.dataframe(df_display.style.apply(style_row, axis=1), 
                                     height=300, hide_index=True, use_container_width=True)
            
            # Stats
            total_trades = len(df_log)
            winning = len(df_log[df_log['PnL'] > 0])
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
            
            stats_placeholder.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                <p style="color: #8b9dc3;"><strong>Trades:</strong> {total_trades}</p>
                <p style="color: #8b9dc3;"><strong>Win Rate:</strong> {win_rate:.1f}%</p>
                <p style="color: #8b9dc3;"><strong>Winners:</strong> {winning} | <strong>Losers:</strong> {total_trades - winning}</p>
                <p style="color: #8b9dc3;"><strong>Total P&L:</strong> ₹{tick['total_pnl']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            log_placeholder.info("📭 No trades yet")
            stats_placeholder.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                <p style="color: #8b9dc3;">Waiting for first trade...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Time Compression Logic
        if time_compression and last_tick_timestamp:
            elapsed_real = (tick['ts'] - last_tick_timestamp).total_seconds()
            sleep_duration = min(elapsed_real / 60, 1.0)  # Max 1 second per minute
            time.sleep(sleep_duration)
        elif sleep_time > 0:
            time.sleep(sleep_time)
        
        last_tick_timestamp = tick['ts']
    
    # Final Summary
    st.markdown("---")
    st.markdown("### 🎯 SESSION COMPLETE")
    
    if tick['log']:
        final_df = pd.DataFrame(tick['log'])
        total_trades = len(final_df)
        winning = len(final_df[final_df['PnL'] > 0])
        losing = total_trades - winning
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = final_df['PnL'].sum()
        avg_win = final_df[final_df['PnL'] > 0]['PnL'].mean() if winning > 0 else 0
        avg_loss = final_df[final_df['PnL'] < 0]['PnL'].mean() if losing > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎲 Total Trades", total_trades)
        col2.metric("📊 Win Rate", f"{win_rate:.1f}%")
        col3.metric("💰 Total P&L", f"₹{total_pnl:,.0f}")
        col4.metric("⚡ Avg Win/Loss", f"₹{avg_win:.0f} / ₹{avg_loss:.0f}")
        
        st.success(f"✅ Simulation completed! Final P&L: ₹{total_pnl:,.0f}")
    else:
        st.warning("⚠️ No trades executed during this session")

if __name__ == "__main__":
    main()