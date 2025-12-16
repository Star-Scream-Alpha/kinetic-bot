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
st.set_page_config(page_title="Kinetic Hunter", layout="wide", page_icon="⚡")

BUCKET = "live-market-data"
SYMBOL = "NIFTY"

# Date Range for Simulation (Restored)
START_DATE = datetime.date(2025, 9, 1)
END_DATE = datetime.date(2025, 11, 26)

# Default Visual Date
DEFAULT_DATE = datetime.date(2025, 11, 25)

# Strategy Params
KINETIC_THRESHOLD = 37500     
HVG_WINDOW = 50               
DEGREE_THRESHOLD = 3.5        

STOP_LOSS_POINTS = 20.0       
TAKE_PROFIT_POINTS = 60.0     
MAX_HOLD_SECONDS = 1800       
COOLDOWN_SECONDS = 300        
COST_PER_TRADE = 1.0          

CAPITAL = 1000000             
MARGIN_PER_LOT = 150000       
LOT_SIZE = 75
NUM_LOTS = int(CAPITAL / MARGIN_PER_LOT) 
QTY = NUM_LOTS * LOT_SIZE     

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
    def __init__(self):
        self.tick_buffer = deque(maxlen=HVG_WINDOW) 
        self.last_score = 0
        self.last_degree = 0.0

    def process_tick(self, ltp, cumulative_volume):
        self.tick_buffer.append([ltp, cumulative_volume])
        
        if len(self.tick_buffer) < HVG_WINDOW: return 0 
        
        data = np.array(self.tick_buffer)
        prices = data[:, 0]
        vols = data[:, 1]
        
        vol_diff = np.diff(vols)
        trade_vol = np.where(vol_diff > 0, vol_diff, 0)
        displacement = abs(prices[-1] - prices[0])
        kinetic_score = np.sum(trade_vol) / (displacement + 0.05)
        self.last_score = kinetic_score
        
        if kinetic_score > KINETIC_THRESHOLD:
            mean_degree = calculate_hvg_mean_degree(prices)
            self.last_degree = mean_degree
            
            if mean_degree < DEGREE_THRESHOLD:
                # --- INVERSE LOGIC APPLIED ---
                # Original: direction = 1 if prices[-1] > prices[0] else -1
                # New (Fade):
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
        
        # --- FIXED DATE PARSING ---
        if 'DateTime' not in df.columns:
            # Explicit string conversion to ensure format consistency
            # Combine Date (e.g., "01/09/2025") and Time (e.g., "09:15:00.123")
            # dayfirst=True handles DD/MM/YYYY correctly
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
# 4. EXECUTION ENGINE (Generator for Streamlit)
# ==========================================
def run_simulation(df, update_speed):
    brain = KineticGraphBrain()
    state = 'SEARCHING'
    
    entry_price = 0.0
    entry_time = None
    direction = 0 
    cooldown_start = None
    
    daily_pnl = 0.0
    trades_log = []
    
    # Yield initial state
    yield {
        "ts": df.iloc[0]['DateTime'], "ltp": df.iloc[0]['LTP'], 
        "score": 0, "degree": 0, "state": state, 
        "pnl": 0, "total_pnl": 0, "dir": 0, "log": []
    }
    
    tick_count = 0
    
    for row in df.itertuples():
        ts = row.DateTime
        ltp = row.LTP
        vol = row.Volume
        tick_count += 1
        
        # Process Logic
        if state == 'COOLDOWN':
            if (ts - cooldown_start).total_seconds() > COOLDOWN_SECONDS:
                state = 'SEARCHING'
                brain.tick_buffer.clear()
        
        elif state == 'SEARCHING':
            sig = brain.process_tick(ltp, vol)
            if sig != 0:
                state = 'IN_TRADE'
                direction = sig
                entry_price = ltp
                entry_time = ts
                
        elif state == 'IN_TRADE':
            elapsed = (ts - entry_time).total_seconds()
            pnl_pts = 0
            exit_reason = None
            
            if direction == 1: pnl_pts = ltp - entry_price
            else: pnl_pts = entry_price - ltp
            
            if pnl_pts <= -STOP_LOSS_POINTS: exit_reason = "SL"
            elif pnl_pts >= TAKE_PROFIT_POINTS: exit_reason = "TP"
            elif elapsed >= MAX_HOLD_SECONDS: exit_reason = "TIME"
            
            if exit_reason:
                net_pts = pnl_pts - COST_PER_TRADE
                realized_pnl = net_pts * QTY
                daily_pnl += realized_pnl
                
                trades_log.insert(0, { # Prepend for latest first
                    "Time": ts.strftime("%H:%M:%S"),
                    "Type": "LONG" if direction == 1 else "SHORT",
                    "PnL": f"₹{realized_pnl:.0f}",
                    "Reason": exit_reason
                })
                
                state = 'COOLDOWN'
                cooldown_start = ts
                direction = 0
                entry_price = 0.0

        # Calculate Floating PnL for Display
        float_pnl = 0.0
        if state == 'IN_TRADE':
            if direction == 1: float_pnl = (ltp - entry_price) * QTY
            else: float_pnl = (entry_price - ltp) * QTY

        # Yield Update
        # To keep UI responsive, don't yield every tick.
        if tick_count % 20 == 0 or state == 'IN_TRADE':
             yield {
                "ts": ts, "ltp": ltp, 
                "score": brain.last_score, "degree": brain.last_degree, 
                "state": state, "pnl": float_pnl, 
                "total_pnl": daily_pnl, "dir": direction,
                "log": trades_log
            }
            
# ==========================================
# 5. STREAMLIT UI
# ==========================================
def main():
    st.title("⚡ Kinetic Hunter Dashboard")
    
    # --- SIDEBAR ---
    st.sidebar.header("Simulation Config")
    
    selected_date = st.sidebar.date_input("Select Date", DEFAULT_DATE, min_value=START_DATE, max_value=END_DATE)
    speed = st.sidebar.select_slider("Replay Speed", options=["Slow", "Normal", "Fast", "Hyper"], value="Fast")
    
    speed_map = {"Slow": 0.1, "Normal": 0.05, "Fast": 0.01, "Hyper": 0.001}
    sleep_time = speed_map[speed]
    
    if st.sidebar.button("🚀 Start Simulation", type="primary"):
        run_dashboard(selected_date, sleep_time)

def run_dashboard(date_obj, sleep_time):
    # 1. Load Data
    with st.spinner(f"Loading Tick Data for {date_obj}..."):
        df = get_data_for_date(date_obj)
    
    if df is None or df.empty:
        st.error("No data found for selected date.")
        return
    
    # 2. Setup Layout
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1_placeholder = kpi1.empty()
    kpi2_placeholder = kpi2.empty()
    kpi3_placeholder = kpi3.empty()
    kpi4_placeholder = kpi4.empty()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Analysis")
        chart_placeholder = st.empty()
        
        st.write("### Kinetic Energy & Topology")
        bar1_label = st.empty()
        bar1 = st.progress(0)
        bar2_label = st.empty()
        bar2 = st.progress(0)

    with col2:
        st.subheader("Trade Log")
        log_placeholder = st.empty()

    # 3. Run Loop
    sim = run_simulation(df, sleep_time)
    
    for tick in sim:
        # Update Metrics
        ts_str = tick['ts'].strftime("%H:%M:%S")
        
        kpi1_placeholder.metric("Time", ts_str)
        kpi2_placeholder.metric("LTP", f"{tick['ltp']:.2f}")
        
        # Dynamic Color for PnL
        pnl_val = tick['total_pnl'] + tick['pnl']
        kpi3_placeholder.metric("Total PnL", f"₹{pnl_val:,.0f}", delta=f"{tick['pnl']:.0f}" if tick['pnl'] != 0 else None)
        
        state_label = tick['state']
        if tick['state'] == 'IN_TRADE':
            dir_str = "LONG" if tick['dir'] == 1 else "SHORT"
            state_label = f"🟢 {dir_str}"
        
        kpi4_placeholder.metric("Status", state_label)
        
        # Update Bars
        # Kinetic (Log scale approx)
        k_norm = min(tick['score'] / 150000, 1.0)
        bar1_label.text(f"Kinetic Score: {tick['score']:.0f} (Threshold: {KINETIC_THRESHOLD})")
        bar1.progress(k_norm)
        
        # Degree (Inverted: 3.5 is threshold, lower is better)
        # Scale: 0 to 5.
        d_norm = 0.0
        if tick['degree'] > 0:
            d_val = max(0, 6.0 - tick['degree']) # Invert so higher bar = better signal (lower degree)
            d_norm = min(d_val / 3.0, 1.0)
            
        bar2_label.text(f"Graph Fragility (Degree): {tick['degree']:.2f} (Break < {DEGREE_THRESHOLD})")
        bar2.progress(d_norm)
        
        # Update Log
        if tick['log']:
            log_df = pd.DataFrame(tick['log'])
            log_placeholder.dataframe(log_df, height=400, hide_index=True)
        else:
            log_placeholder.info("Waiting for signals...")
            
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()