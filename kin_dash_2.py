import boto3
import pandas as pd
import numpy as np
import datetime
import time
from collections import deque
from io import BytesIO
from numba import njit
import streamlit as st

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Kinetic Hunter Pro", layout="wide", page_icon="⚡")

# Enhanced Custom CSS with dark theme and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
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
        margin-bottom: 0 !important;
    }
    
    h3 {
        color: #8b9dc3;
        font-weight: 300;
        font-style: italic;
        margin-top: 0 !important;
    }
    
    h5 {
        color: #667eea;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 15px 0 rgba(17, 153, 142, 0.4);
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
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .status-cooldown {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 15px 0 rgba(245, 87, 108, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .dataframe {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #667eea !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }
    
    .dataframe td {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #667eea;
        font-weight: 700;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    .trade-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .caption-text {
        color: #8b9dc3;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    
    .stNumberInput label {
        color: #8b9dc3 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

BUCKET = "live-market-data"
SYMBOL = "NIFTY"

# Date Range for Simulation
START_DATE = datetime.date(2025, 9, 1)
END_DATE = datetime.date(2025, 11, 26)

# Default Visual Date
DEFAULT_DATE = datetime.date(2025, 11, 25)

# Market Hours
MARKET_START = datetime.time(9, 15)
MARKET_END = datetime.time(15, 30)

# Contract Dates
EXPIRY_SEP = datetime.date(2025, 9, 25)
EXPIRY_OCT = datetime.date(2025, 10, 30)
EXPIRY_NOV = datetime.date(2025, 11, 27)

# ==========================================
# 1. NUMBA GRAPH ENGINE (HVG)
# ==========================================
@njit
def calculate_hvg_mean_degree(prices):
    n = len(prices)
    if n < 2: return 0.0
    
    total_degree = 0
    for i in range(n):
        current_degree = 0
        # Backwards
        if i > 0:
            max_h = -99999999.0
            for j in range(i - 1, -1, -1):
                if j == i - 1: 
                    current_degree += 1; max_h = prices[j]
                elif prices[j] > max_h:
                    current_degree += 1; max_h = prices[j]
                if prices[j] >= prices[i]: break
        # Forwards
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
# 2. KINETIC GRAPH BRAIN
# ==========================================
class KineticGraphBrain:
    def __init__(self, hvg_window, kinetic_threshold, degree_threshold):
        self.hvg_window = hvg_window
        self.kinetic_threshold = kinetic_threshold
        self.degree_threshold = degree_threshold
        self.tick_buffer = deque(maxlen=hvg_window) 
        self.last_score = 0
        self.last_degree = 0.0

    def process_tick(self, ltp, cumulative_volume):
        self.tick_buffer.append([ltp, cumulative_volume])
        
        if len(self.tick_buffer) < self.hvg_window: return 0 
        
        data = np.array(self.tick_buffer)
        prices = data[:, 0]
        vols = data[:, 1]
        
        vol_diff = np.diff(vols)
        trade_vol = np.where(vol_diff > 0, vol_diff, 0)
        displacement = abs(prices[-1] - prices[0])
        kinetic_score = np.sum(trade_vol) / (displacement + 0.05)
        self.last_score = kinetic_score
        
        if kinetic_score > self.kinetic_threshold:
            mean_degree = calculate_hvg_mean_degree(prices)
            self.last_degree = mean_degree
            
            if mean_degree < self.degree_threshold:
                # Inverse Logic: Break UP -> SHORT, Break DOWN -> LONG
                direction = -1 if prices[-1] > prices[0] else 1
                return direction
                
        return 0

# ==========================================
# 3. DATA LOADING (Cached)
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
            try:
                df['DateTime'] = pd.to_datetime(
                    df['Date'].astype(str) + " " + df['Time'].astype(str), 
                    dayfirst=True, 
                    errors='coerce'
                )
            except Exception as e:
                st.error(f"Date parsing failed: {e}")
                return None

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
# 4. EXECUTION ENGINE (Generator)
# ==========================================
def run_simulation(df, params):
    brain = KineticGraphBrain(
        params['hvg_window'],
        params['kinetic_threshold'],
        params['degree_threshold']
    )
    
    state = 'SEARCHING'
    
    entry_price = 0.0
    entry_time = None
    direction = 0 
    cooldown_start = None
    
    daily_pnl = 0.0
    trades_log = []
    
    tick_count = 0
    
    # Yield initial state
    yield {
        "ts": df.iloc[0]['DateTime'], "ltp": df.iloc[0]['LTP'], 
        "score": 0, "degree": 0, "state": state, 
        "pnl": 0, "total_pnl": 0, "dir": 0, "log": [],
        "entry_price": 0
    }

    for row in df.itertuples():
        ts = row.DateTime
        ltp = row.LTP
        vol = row.Volume
        curr_time = ts.time()
        
        tick_count += 1
        exit_reason = None
        
        # --- LOGIC ---
        
        # 1. COOLDOWN STATE
        if state == 'COOLDOWN':
            if (ts - cooldown_start).total_seconds() > params['cooldown_seconds']:
                state = 'SEARCHING'
                brain.tick_buffer.clear()
        
        # 2. SEARCHING STATE
        elif state == 'SEARCHING':
            sig = brain.process_tick(ltp, vol)
            
            # FILTER: TIME (09:15 - 15:30)
            is_market_open = (curr_time >= MARKET_START) and (curr_time < MARKET_END)
            
            if sig != 0 and is_market_open:
                state = 'IN_TRADE'
                direction = sig
                entry_price = ltp
                entry_time = ts
                
        # 3. IN TRADE STATE
        elif state == 'IN_TRADE':
            elapsed = (ts - entry_time).total_seconds()
            pnl_pts = 0
            
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

        # Calculate Floating PnL
        float_pnl = 0.0
        if state == 'IN_TRADE':
            if direction == 1: float_pnl = (ltp - entry_price) * params['qty']
            else: float_pnl = (entry_price - ltp) * params['qty']

        # Yield Update - More frequent updates
        should_yield = (
            tick_count % 10 == 0 or 
            state == 'IN_TRADE' or 
            exit_reason is not None or
            state == 'COOLDOWN'
        )
        
        if should_yield:
             yield {
                "ts": ts, "ltp": ltp, 
                "score": brain.last_score, "degree": brain.last_degree, 
                "state": state, "pnl": float_pnl, 
                "total_pnl": daily_pnl, "dir": direction,
                "log": trades_log, "entry_price": entry_price
            }
            
# ==========================================
# 5. STREAMLIT UI
# ==========================================
def main():
    st.title("⚡ Kinetic Hunter Pro")
    st.markdown("### *Graph-Based Volatility Arbitrage System*")
    
    st.markdown("---")
    
    # --- SIDEBAR ---
    st.sidebar.header("🎯 Simulation Control")
    st.sidebar.markdown("---")
    
    selected_date = st.sidebar.date_input(
        "📅 Select Trading Date", 
        DEFAULT_DATE, 
        min_value=START_DATE, 
        max_value=END_DATE
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Replay Speed")
    
    speed = st.sidebar.select_slider(
        "⚡ Replay Speed", 
        options=["Realtime", "Fast", "Hyper", "Warp", "Instant"], 
        value="Fast",
        help="Speed of simulation playback"
    )
    speed_map = {"Realtime": 0.1, "Fast": 0.01, "Hyper": 0.001, "Warp": 0.0001, "Instant": 0}
    sleep_time = speed_map[speed]
    
    time_compression_enabled = st.sidebar.checkbox(
        "⏰ Enable Time Compression",
        value=False,
        help="1 second real time = 1 minute market time"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Strategy Parameters")
    
    # Editable Parameters
    kinetic_threshold = st.sidebar.number_input(
        "Kinetic Threshold",
        min_value=1000.0,
        max_value=100000.0,
        value=37500.0,
        step=500.0,
        help="Energy threshold to trigger graph analysis"
    )
    
    hvg_window = st.sidebar.number_input(
        "HVG Window",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Number of ticks for visibility graph"
    )
    
    degree_threshold = st.sidebar.number_input(
        "Degree Threshold",
        min_value=0.5,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Graph fragility breakpoint"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💼 Risk Management")
    
    stop_loss_points = st.sidebar.number_input(
        "Stop Loss (points)",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=5.0
    )
    
    take_profit_points = st.sidebar.number_input(
        "Take Profit (points)",
        min_value=10.0,
        max_value=200.0,
        value=60.0,
        step=5.0
    )
    
    max_hold_minutes = st.sidebar.number_input(
        "Max Hold Time (minutes)",
        min_value=1,
        max_value=120,
        value=30,
        step=5
    )
    
    cooldown_minutes = st.sidebar.number_input(
        "Cooldown Period (minutes)",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Position Sizing")
    
    qty = st.sidebar.number_input(
        "Quantity (units)",
        min_value=1,
        max_value=10000,
        value=450,
        step=25,
        help="Total contract quantity"
    )
    
    cost_per_trade = st.sidebar.number_input(
        "Cost Per Trade (points)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Brokerage + slippage"
    )
    
    st.sidebar.markdown("---")
    
    # Create params dict
    params = {
        'kinetic_threshold': kinetic_threshold,
        'hvg_window': hvg_window,
        'degree_threshold': degree_threshold,
        'stop_loss_points': stop_loss_points,
        'take_profit_points': take_profit_points,
        'max_hold_seconds': max_hold_minutes * 60,
        'cooldown_seconds': cooldown_minutes * 60,
        'qty': qty,
        'cost_per_trade': cost_per_trade
    }
    
    if st.sidebar.button("🚀 Launch Simulation", type="primary", use_container_width=True):
        run_dashboard(selected_date, sleep_time, time_compression_enabled, params)

def run_dashboard(date_obj, sleep_time, time_compression_enabled, params):
    with st.spinner(f"⏳ Loading Tick Data for {date_obj}..."):
        df = get_data_for_date(date_obj)
    
    if df is None or df.empty:
        st.error("❌ No data found for selected date.")
        return
    
    st.success(f"✅ Loaded {len(df):,} ticks")
    
    # Display speed info
    if time_compression_enabled:
        st.info("⏰ Time Compression: 1 minute of market time = 1 second real time")
    else:
        st.info("⚡ Running in normal speed mode")
    
    # --- LAYOUT SETUP ---
    
    # Top KPI Row
    st.markdown("### 📈 Live Dashboard")
    kpi_cols = st.columns(5)
    time_metric = kpi_cols[0].empty()
    price_metric = kpi_cols[1].empty()
    pnl_metric = kpi_cols[2].empty()
    status_metric = kpi_cols[3].empty()
    entry_metric = kpi_cols[4].empty()
    
    # Status Banner
    status_banner = st.empty()
    
    st.markdown("---")
    
    # Main Content
    col_left, col_right = st.columns([2.5, 1.5])
    
    with col_left:
        st.markdown("##### 📊 MARKET VITALS")
        
        # Kinetic Chart
        st.markdown(f'<div class="caption-text">⚡ Kinetic Energy Score (Trigger > {params["kinetic_threshold"]:,.0f})</div>', unsafe_allow_html=True)
        chart_kinetic = st.empty()
        
        # Degree Chart
        st.markdown(f'<div class="caption-text">🔗 Graph Fragility Index (Break < {params["degree_threshold"]})</div>', unsafe_allow_html=True)
        chart_degree = st.empty()

    with col_right:
        st.markdown("##### 📜 TRADE JOURNAL")
        log_placeholder = st.empty()
        
        st.markdown("---")
        st.markdown("##### 📊 SESSION STATS")
        stats_placeholder = st.empty()
        
    # --- RUN SIMULATION ---
    sim = run_simulation(df, params)
    
    # Data Buffers
    kinetic_hist = deque(maxlen=100)
    degree_hist = deque(maxlen=100)
    
    # For time-based compression
    last_tick_timestamp = None
    
    for tick in sim:
        # 1. Update Header Metrics
        time_metric.metric("⏰ TIME", tick['ts'].strftime("%H:%M:%S"))
        price_metric.metric("💹 NIFTY FUT", f"₹{tick['ltp']:.2f}")
        
        total_val = tick['total_pnl'] + tick['pnl']
        delta_str = f"₹{tick['pnl']:.0f}" if tick['pnl'] != 0 else None
        pnl_metric.metric("💰 NET P&L", f"₹{total_val:,.0f}", delta=delta_str)
        
        # Show entry price when in trade
        if tick['entry_price'] > 0:
            entry_metric.metric("🎯 ENTRY", f"₹{tick['entry_price']:.2f}")
        else:
            entry_metric.metric("🎯 ENTRY", "-")
        
        # 2. Dynamic Status Banner
        s = tick['state']
        if s == 'SEARCHING':
            status_metric.metric("🎯 STATUS", "SCANNING")
            status_banner.markdown('<div class="status-scanning">📡 SCANNING FOR KINETIC BURSTS...</div>', unsafe_allow_html=True)
        elif s == 'IN_TRADE':
            d_str = "LONG 🟢" if tick['dir'] == 1 else "SHORT 🔴"
            status_metric.metric("🎯 STATUS", "ACTIVE")
            status_banner.markdown(f'<div class="status-active">⚡ TRADE ACTIVE: {d_str} | Floating P&L: ₹{tick["pnl"]:.0f}</div>', unsafe_allow_html=True)
        elif s == 'COOLDOWN':
            status_metric.metric("🎯 STATUS", "COOLDOWN")
            status_banner.markdown('<div class="status-cooldown">❄️ COOLDOWN MODE - RESETTING GRAPH ENGINE</div>', unsafe_allow_html=True)

        # 3. Update Charts
        kinetic_hist.append(tick['score'])
        d_val = tick['degree'] if tick['degree'] > 0 else None
        degree_hist.append(d_val)
        
        # Create dataframes for proper chart updates
        df_kinetic = pd.DataFrame(list(kinetic_hist), columns=['Kinetic Score'])
        df_degree = pd.DataFrame(list(degree_hist), columns=['Degree'])
        
        chart_kinetic.line_chart(df_kinetic, height=220, use_container_width=True)
        chart_degree.line_chart(df_degree, height=220, use_container_width=True)

        # 4. Update Log
        try:
            if tick['log']:
                df_log = pd.DataFrame(tick['log'])
                
                if len(df_log) > 0:
                    # Format PnL column with currency
                    df_log['PnL Display'] = df_log['PnL'].apply(lambda x: f"₹{x:,.0f}")
                    df_log['Pts Display'] = df_log['Pts'].apply(lambda x: f"{x:.1f}")
                    
                    # Create display dataframe
                    df_display = df_log[['Entry Time', 'Exit Time', 'Type', 'Entry', 'Exit']].copy()
                    df_display['PnL'] = df_log['PnL Display']
                    df_display['Pts'] = df_log['Pts Display']
                    df_display['Reason'] = df_log['Reason']
                    
                    # Style based on actual PnL value
                    def style_row(row):
                        pnl_val = df_log.loc[row.name, 'PnL']
                        if pnl_val > 0:
                            return ['background-color: rgba(17, 153, 142, 0.2); color: #38ef7d'] * len(row)
                        else:
                            return ['background-color: rgba(245, 87, 108, 0.2); color: #f5576c'] * len(row)
                    
                    styled_df = df_display.style.apply(style_row, axis=1)
                    
                    log_placeholder.dataframe(
                        styled_df,
                        height=300, 
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Session Stats
                    total_trades = len(df_log)
                    winning_trades = len(df_log[df_log['PnL'] > 0])
                    losing_trades = total_trades - winning_trades
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    avg_pnl = df_log['PnL'].mean() if total_trades > 0 else 0
                    total_pnl_calc = df_log['PnL'].sum()
                    
                    stats_placeholder.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
                        <p style="color: #8b9dc3; margin: 5px 0;"><strong>Total Trades:</strong> {total_trades}</p>
                        <p style="color: #38ef7d; margin: 5px 0;"><strong>Winning:</strong> {winning_trades}</p>
                        <p style="color: #f5576c; margin: 5px 0;"><strong>Losing:</strong> {losing_trades}</p>
                        <p style="color: #667eea; margin: 5px 0;"><strong>Win Rate:</strong> {win_rate:.1f}%</p>
                        <p style="color: #8b9dc3; margin: 5px 0;"><strong>Avg P&L:</strong> ₹{avg_pnl:,.0f}</p>
                        <p style="color: #8b9dc3; margin: 5px 0;"><strong>Total P&L:</strong> ₹{total_pnl_calc:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Empty log list
                    log_placeholder.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); padding: 40px; border-radius: 10px; text-align: center; color: #8b9dc3;">
                        <p style="font-size: 1.2rem;">📭</p>
                        <p>No completed trades yet...</p>
                        <p style="font-size: 0.9rem; color: #667eea;">Current State: {tick['state']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_placeholder.markdown("""
                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
                        <p style="color: #8b9dc3; margin: 5px 0;">Waiting for first completed trade...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # No log at all
                log_placeholder.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 40px; border-radius: 10px; text-align: center; color: #8b9dc3;">
                    <p style="font-size: 1.2rem;">📭</p>
                    <p>No completed trades yet...</p>
                    <p style="font-size: 0.9rem; color: #667eea;">Current State: {tick['state']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                stats_placeholder.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
                    <p style="color: #8b9dc3; margin: 5px 0;">Waiting for first completed trade...</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            log_placeholder.error(f"Error displaying trades: {e}")
            st.write("Debug - Trade log:", tick['log'])
            
        # Apply speed control
        if time_compression_enabled and last_tick_timestamp:
            # Time compression: 1 minute market time = 1 second real time
            current_tick_time = tick['ts']
            market_time_passed = (current_tick_time - last_tick_timestamp).total_seconds()
            real_sleep_time = market_time_passed / 60.0  # 60 seconds market time = 1 second real time
            if real_sleep_time > 0:
                time.sleep(real_sleep_time)
        elif sleep_time > 0:
            # Normal tick-based speed
            time.sleep(sleep_time)
        
        # Update timestamp for next iteration
        last_tick_timestamp = tick['ts']

if __name__ == "__main__":
    main()