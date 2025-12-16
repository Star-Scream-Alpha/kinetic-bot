import boto3
import pandas as pd
import numpy as np
import datetime
import time
import os
from pathlib import Path
from collections import deque
from io import BytesIO
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
BUCKET = "live-market-data"
SYMBOL = "NIFTY"
CACHE_DIR = Path("./market_data_cache")  # Local folder to store S3 files

# Backtest Range
START_DATE = datetime.date(2025, 9, 1)
END_DATE = datetime.date(2025, 11, 26)

# Market Hours
MARKET_START = datetime.time(9, 15)
MARKET_END = datetime.time(15, 30)

# Contract Dates
EXPIRY_SEP = datetime.date(2025, 9, 25)
EXPIRY_OCT = datetime.date(2025, 10, 30)
EXPIRY_NOV = datetime.date(2025, 11, 27)

# Strategy Parameters
PARAMS = {
    'kinetic_threshold': 140000.0,
    'hvg_window': 50,
    'degree_threshold': 3.5,
    'stop_loss_points': 90.0,
    'take_profit_points': 60.0, # Target to hit
    
    # NEW: Trailing Stop Params
    'use_trailing_stop': True,
    'trail_trigger': 40.0,      # Start trailing after 40 pts profit
    'trail_gap': 15.0,          # Keep SL 15 pts behind peak
    
    'max_hold_seconds': 30 * 60,
    'cooldown_seconds': 5 * 60,
    'qty': 450,
    'cost_per_trade': 1.0
}

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
# ==========================================
# 2. UPGRADED KINETIC BRAIN (TREND FOLLOWING)
# ==========================================
# ==========================================
# 2. KINETIC BRAIN (INVERTED / MEAN REVERSION)
# ==========================================
class KineticGraphBrain:
    def __init__(self, hvg_window, kinetic_threshold, degree_threshold):
        self.hvg_window = hvg_window
        self.kinetic_threshold = kinetic_threshold
        self.degree_threshold = degree_threshold
        self.tick_buffer = deque(maxlen=hvg_window) 

    def process_tick(self, ltp, cumulative_volume):
        self.tick_buffer.append([ltp, cumulative_volume])
        
        if len(self.tick_buffer) < self.hvg_window: return 0 
        
        data = np.array(self.tick_buffer)
        prices = data[:, 0]
        vols = data[:, 1]
        
        # Kinetic Energy Calc
        vol_diff = np.diff(vols)
        trade_vol = np.where(vol_diff > 0, vol_diff, 0)
        displacement = abs(prices[-1] - prices[0])
        if displacement == 0: displacement = 0.05   
        kinetic_score = np.sum(trade_vol) / displacement
        
        if kinetic_score > self.kinetic_threshold:
            mean_degree = calculate_hvg_mean_degree(prices)
            
            if mean_degree < self.degree_threshold:
                
                # --- THE INVERSION ---
                # Old Logic: Price Up -> Buy
                # New Logic: Price Up -> Sell (Fade the move)
                
                if prices[-1] > prices[0]:
                    return -1 # SELL (Fade the pump)
                else:
                    return 1  # BUY (Catch the knife)
                
        return 0

# ==========================================
# 4. TRADING ENGINE (INVERTED FILTERS)
# ==========================================
def run_single_day(date_obj):
    df = get_data_for_date(date_obj)
    if df is None or df.empty: return []

    df = process_dataframe(df)
    
    # VWAP Calc
    df['TickVol'] = df['Volume'].diff().fillna(0).clip(lower=0)
    df['PV'] = df['LTP'] * df['TickVol']
    df['CumPV'] = df['PV'].cumsum()
    df['CumVol'] = df['TickVol'].cumsum()
    df['VWAP'] = df['CumPV'] / df['CumVol']
    
    brain = KineticGraphBrain(PARAMS['hvg_window'], PARAMS['kinetic_threshold'], PARAMS['degree_threshold'])
    
    state = 'SEARCHING'
    entry_price = 0.0
    entry_time = None
    direction = 0 
    cooldown_start = None
    highest_pnl_pts = -999.0
    daily_trades = []
    
    # Keep Lunch Filter (It saves money in both directions)
    LUNCH_START = datetime.time(11, 30)
    LUNCH_END = datetime.time(13, 00)
    
    for row in df.itertuples():
        ts = row.DateTime
        ltp = row.LTP
        vol = row.Volume
        vwap = row.VWAP
        curr_time = ts.time()
        
        if LUNCH_START <= curr_time <= LUNCH_END:
             if state == 'SEARCHING': continue

        if state == 'COOLDOWN':
            if (ts - cooldown_start).total_seconds() > PARAMS['cooldown_seconds']:
                state = 'SEARCHING'
                brain.tick_buffer.clear()
        
        elif state == 'SEARCHING':
            sig = brain.process_tick(ltp, vol)
            
            if sig != 0 and (MARKET_START <= curr_time < MARKET_END):
                
                valid_trade = False
                
                # --- INVERTED VWAP REGIME ---
                # We want to SHORT when price is expensive (Above VWAP)
                # We want to LONG when price is cheap (Below VWAP)
                
                if sig == 1: # Brain says BUY (because price crashed)
                    if ltp < vwap: # Only buy if we are "Cheap"
                        valid_trade = True
                        
                elif sig == -1: # Brain says SELL (because price pumped)
                    if ltp > vwap: # Only sell if we are "Expensive"
                        valid_trade = True
                
                if valid_trade:
                    state = 'IN_TRADE'
                    direction = sig
                    entry_price = ltp
                    entry_time = ts
                    highest_pnl_pts = -999.0 
                
        elif state == 'IN_TRADE':
            elapsed = (ts - entry_time).total_seconds()
            
            if direction == 1: current_pts = ltp - entry_price
            else: current_pts = entry_price - ltp
            
            if current_pts > highest_pnl_pts: highest_pnl_pts = current_pts
            
            exit_reason = None
            
            if current_pts <= -PARAMS['stop_loss_points']: exit_reason = "SL"
            elif elapsed >= PARAMS['max_hold_seconds']: exit_reason = "TIME"
            elif curr_time >= MARKET_END: exit_reason = "EOD"
            else:
                if PARAMS['use_trailing_stop']:
                    if highest_pnl_pts >= PARAMS['trail_trigger']:
                        if current_pts <= (highest_pnl_pts - PARAMS['trail_gap']):
                            exit_reason = "TRAIL_SL"
                else:
                    if current_pts >= PARAMS['take_profit_points']: exit_reason = "TP"

            # VWAP Reversion Target (Optional but smart for mean reversion)
            # If we were Shorting above VWAP, and we touch VWAP, take profit.
            # Uncomment next 2 lines if you want this:
            # if direction == -1 and ltp <= vwap: exit_reason = "VWAP_TARGET"
            # if direction == 1 and ltp >= vwap: exit_reason = "VWAP_TARGET"

            if exit_reason:
                net_pts = current_pts - PARAMS['cost_per_trade']
                realized_pnl = net_pts * PARAMS['qty']
                
                daily_trades.append({
                    "Date": ts.date(), "Entry Time": entry_time, "Exit Time": ts,
                    "Type": "LONG" if direction == 1 else "SHORT",
                    "Entry Price": entry_price, "Exit Price": ltp,
                    "Net Points": net_pts, "PnL": realized_pnl, "Reason": exit_reason
                })
                state = 'COOLDOWN'
                cooldown_start = ts

    return daily_trades
# ==========================================
# 4. TRADING ENGINE (WITH VWAP & TIME FILTER)
# ==========================================
def run_single_day(date_obj):
    df = get_data_for_date(date_obj)
    if df is None or df.empty: return []

    df = process_dataframe(df)
    
    # --- CALCULATE INTRADAY VWAP ---
    # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    # Note: We need 'Tick Volume' (change in cumulative volume)
    df['TickVol'] = df['Volume'].diff().fillna(0)
    df['TickVol'] = df['TickVol'].clip(lower=0) # Remove negative noise
    df['PV'] = df['LTP'] * df['TickVol']
    
    # Reset VWAP every day (The dataframe is already 1 day, so just cumsum)
    df['CumPV'] = df['PV'].cumsum()
    df['CumVol'] = df['TickVol'].cumsum()
    df['VWAP'] = df['CumPV'] / df['CumVol']
    
    brain = KineticGraphBrain(PARAMS['hvg_window'], PARAMS['kinetic_threshold'], PARAMS['degree_threshold'])
    
    state = 'SEARCHING'
    entry_price = 0.0
    entry_time = None
    direction = 0 
    cooldown_start = None
    highest_pnl_pts = -999.0
    daily_trades = []
    
    # Lunch Time Filter (Avoid 11:30 - 13:00)
    LUNCH_START = datetime.time(11, 30)
    LUNCH_END = datetime.time(13, 00)
    
    for row in df.itertuples():
        ts = row.DateTime
        ltp = row.LTP
        vol = row.Volume
        vwap = row.VWAP
        curr_time = ts.time()
        
        # Skip Lunch Chop
        if LUNCH_START <= curr_time <= LUNCH_END:
            # If in trade, let it run, but don't take new trades
            if state == 'SEARCHING': continue

        # COOLDOWN
        if state == 'COOLDOWN':
            if (ts - cooldown_start).total_seconds() > PARAMS['cooldown_seconds']:
                state = 'SEARCHING'
                brain.tick_buffer.clear()
        
        # SEARCHING
        elif state == 'SEARCHING':
            sig = brain.process_tick(ltp, vol)
            
            if sig != 0 and (MARKET_START <= curr_time < MARKET_END):
                
                # --- VWAP REGIME FILTER (CRITICAL) ---
                # Only Buy if Price is ABOVE VWAP
                # Only Sell if Price is BELOW VWAP
                valid_trade = False
                
                if sig == 1 and ltp > vwap:
                    valid_trade = True
                elif sig == -1 and ltp < vwap:
                    valid_trade = True
                
                if valid_trade:
                    state = 'IN_TRADE'
                    direction = sig
                    entry_price = ltp
                    entry_time = ts
                    highest_pnl_pts = -999.0 
                
        # IN TRADE
        elif state == 'IN_TRADE':
            elapsed = (ts - entry_time).total_seconds()
            
            if direction == 1: current_pts = ltp - entry_price
            else: current_pts = entry_price - ltp
            
            if current_pts > highest_pnl_pts: highest_pnl_pts = current_pts
            
            exit_reason = None
            
            # SL / Time / EOD checks
            if current_pts <= -PARAMS['stop_loss_points']: exit_reason = "SL"
            elif elapsed >= PARAMS['max_hold_seconds']: exit_reason = "TIME"
            elif curr_time >= MARKET_END: exit_reason = "EOD"
            else:
                # Trailing Stop Logic
                if PARAMS['use_trailing_stop']:
                    if highest_pnl_pts >= PARAMS['trail_trigger']:
                        if current_pts <= (highest_pnl_pts - PARAMS['trail_gap']):
                            exit_reason = "TRAIL_SL"
                else:
                    if current_pts >= PARAMS['take_profit_points']: exit_reason = "TP"
            
            # Special Exit: VWAP Cross (If price crosses back over VWAP, bail out)
            if direction == 1 and ltp < vwap: exit_reason = "VWAP_CROSS"
            elif direction == -1 and ltp > vwap: exit_reason = "VWAP_CROSS"

            if exit_reason:
                net_pts = current_pts - PARAMS['cost_per_trade']
                realized_pnl = net_pts * PARAMS['qty']
                
                daily_trades.append({
                    "Date": ts.date(), "Entry Time": entry_time, "Exit Time": ts,
                    "Type": "LONG" if direction == 1 else "SHORT",
                    "Entry Price": entry_price, "Exit Price": ltp,
                    "Net Points": net_pts, "PnL": realized_pnl, "Reason": exit_reason
                })
                state = 'COOLDOWN'
                cooldown_start = ts

    return daily_trades
# ==========================================
# 3. DATA LOADING (WITH CACHING)
# ==========================================
def get_trading_symbol(current_date):
    if current_date <= EXPIRY_SEP: return "NIFTY25SEPFUT"
    elif current_date <= EXPIRY_OCT: return "NIFTY25OCTFUT"
    else: return "NIFTY25NOVFUT"

def get_data_for_date(date_obj):
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    year = date_obj.year; month = date_obj.month; day = date_obj.day
    ts = get_trading_symbol(date_obj)
    filename = f"{year}_{month:02d}_{day:02d}_{SYMBOL}_{ts}.parquet"
    local_path = CACHE_DIR / filename
    
    # 1. Try Local Cache First
    if local_path.exists():
        try:
            return pd.read_parquet(local_path)
        except Exception:
            pass # File corrupt, fallback to S3

    # 2. Download from S3 if not local
    s3 = boto3.client("s3")
    key = f"year={year}/month={month:02d}/day={day:02d}/Futures/{SYMBOL}/{ts}.parquet"
    
    try:
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
        except:
            return None # File not found on S3

        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        
        # Save to cache for next time
        df.to_parquet(local_path)
        return df

    except Exception as e:
        return None

def process_dataframe(df):
    """Standardizes dataframe format."""
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
        
    return df.dropna(subset=['DateTime', 'LTP']).sort_values('DateTime').reset_index(drop=True)

# ==========================================
# 4. TRADING ENGINE (WORKER FUNCTION)
# ==========================================
# ==========================================
# 5. MAIN EXECUTION (PARALLEL)
# ==========================================
def generate_charts(df_res):
    """Generates an equity curve image"""
    df_res = df_res.sort_values('Exit Time')
    df_res['Cumulative PnL'] = df_res['PnL'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['Exit Time'], df_res['Cumulative PnL'], color='#667eea', linewidth=2)
    plt.fill_between(df_res['Exit Time'], df_res['Cumulative PnL'], alpha=0.1, color='#667eea')
    plt.title(f"Cumulative PnL (Net): {START_DATE} to {END_DATE}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylabel("INR Profit")
    
    filename = "backtest_equity_curve.png"
    plt.savefig(filename)
    print(f"📈 Equity curve saved to {filename}")

def main():
    start_time = time.time()
    print(f"🚀 Starting Parallel Backtest: {START_DATE} to {END_DATE}")
    print(f"💾 Caching enabled at: {CACHE_DIR}")
    
    # Generate Date List
    date_list = []
    curr = START_DATE
    while curr <= END_DATE:
        if curr.weekday() < 5: # Skip weekends
            date_list.append(curr)
        curr += datetime.timedelta(days=1)
    
    all_trades = []
    
    # Run Parallel Processing (Uses all available CPU cores)
    # Adjust max_workers if you run out of RAM
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_day, date_list))
    
    # Flatten results
    for day_trades in results:
        all_trades.extend(day_trades)

    # --- REPORTING ---
    elapsed = time.time() - start_time
    print(f"\n✅ Processing Complete in {elapsed:.1f} seconds")
    print("=" * 60)
    
    if not all_trades:
        print("No trades generated.")
        return

    df_res = pd.DataFrame(all_trades)
    
    # Stats Calculation
    total_trades = len(df_res)
    total_pnl = df_res['PnL'].sum()
    avg_pnl = df_res['PnL'].mean()
    win_rate = (len(df_res[df_res['PnL'] > 0]) / total_trades) * 100
    
    # Drawdown
    df_sorted = df_res.sort_values('Exit Time')
    df_sorted['CumPnL'] = df_sorted['PnL'].cumsum()
    peak = df_sorted['CumPnL'].cummax()
    drawdown = df_sorted['CumPnL'] - peak
    max_dd = drawdown.min()

    print(f"Total Trades:     {total_trades}")
    print(f"Net PnL:          ₹{total_pnl:,.2f}")
    print(f"Win Rate:         {win_rate:.2f}%")
    print(f"Avg PnL/Trade:    ₹{avg_pnl:,.2f}")
    print(f"Max Drawdown:     ₹{max_dd:,.2f}")
    print("=" * 60)
    
    # Save Results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    csv_name = f"kinetic_results_{timestamp}.csv"
    df_res.to_csv(csv_name, index=False)
    print(f"💾 Trades saved to: {csv_name}")
    
    # Generate Chart
    generate_charts(df_res)

if __name__ == "__main__":
    main()