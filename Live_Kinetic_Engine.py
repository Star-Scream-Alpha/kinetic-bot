import pandas as pd
import numpy as np
from collections import deque
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
KINETIC_THRESHOLD = 37500
HOLD_SECONDS = 900  # 15 Minutes hold for simulation

# ==========================================
# 1. THE PRODUCTION CLASS (Copy this to your live bot)
# ==========================================
class LiveKineticBrain:
    def __init__(self, threshold=37500):
        """
        Initialize the Kinetic Brain.
        :param threshold: The energy score required to trigger a signal.
        """
        self.threshold = threshold
        # Buffer stores [price, cumulative_volume]
        self.tick_buffer = deque(maxlen=50) 
        self.last_score = 0.0
        self.is_ready = False

    def process_tick(self, ltp, cumulative_volume):
        """
        Feed a single tick into the brain.
        :param ltp: Last Traded Price (float)
        :param cumulative_volume: Total Volume / LTQ Sum / Open Interest (float)
        :return: True if Signal Fired, False otherwise
        """
        # 1. Store Tick
        self.tick_buffer.append([ltp, cumulative_volume])
        
        # 2. Warmup Check
        if len(self.tick_buffer) < 50:
            self.is_ready = False
            return False 
            
        self.is_ready = True

        # 3. Calculate Score
        score = self._calculate_score()
        self.last_score = score
        
        # 4. Check Trigger
        if score > self.threshold:
            return True # SIGNAL FIRED!
            
        return False

    def _calculate_score(self):
        """Internal calculation logic (Vectorized for speed)"""
        data = np.array(self.tick_buffer)
        prices = data[:, 0]
        vols = data[:, 1]
        
        # Calculate Flow (Positive volume deltas only)
        vol_diff = np.diff(vols)
        trade_vol = np.where(vol_diff > 0, vol_diff, 0)
        
        # Calculate Displacement (Price change over 50 ticks)
        displacement = abs(prices[-1] - prices[0])
        
        # Calculate Energy Score
        total_vol = np.sum(trade_vol)
        score = total_vol / (displacement + 0.05)
        
        return score

# ==========================================
# 2. LIVE SIMULATION (MOCK WEBSOCKET)
# ==========================================
def simulate_live_market(csv_file_path):
    print(f"--- STARTING LIVE SIMULATION ON {csv_file_path} ---")
    
    # Load the Mock Websocket Data
    try:
        df = pd.read_csv(csv_file_path)
        
        # Standardize Time Column
        if 'DateTime' not in df.columns:
            if 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Time'], dayfirst=True)
            else:
                # Fallback if no time column exists
                df['DateTime'] = pd.date_range(start='09:15:00', periods=len(df), freq='S')
        else:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
        print(f"Loaded {len(df)} ticks. Starting feed...")
        
    except FileNotFoundError:
        print("Error: File not found.")
        return None

    # Initialize the Brain
    brain = LiveKineticBrain(threshold=KINETIC_THRESHOLD)
    
    trades = []
    in_trade = False
    entry_data = {} # Stores state of current active trade
    
    # --- THE EVENT LOOP ---
    for i, row in df.iterrows():
        # Extract fields
        ltp = row.get('LTP') or row.get('LastTradedPrice') or row.get('Close')
        vol = row.get('Volume') or row.get('LTQ') or row.get('OpenInterest')
        ts = row['DateTime']
        
        if pd.isna(ltp) or pd.isna(vol):
            continue
            
        # 1. FEED THE BRAIN
        # We always feed the brain, even if in trade, to keep buffer fresh
        signal_fired = brain.process_tick(ltp, vol)
        
        # 2. TRADE MANAGEMENT (Sequential - No Overlap)
        if in_trade:
            # Check Exit Condition (Time based for this simulation)
            time_in_trade = (ts - entry_data['Entry_Time']).total_seconds()
            
            if time_in_trade >= HOLD_SECONDS:
                # EXECUTE EXIT
                exit_price = ltp
                pnl = exit_price - entry_data['Entry_Price'] # Long only assumption for magnitude
                
                trades.append({
                    'Entry_Time': entry_data['Entry_Time'],
                    'Entry_Price': entry_data['Entry_Price'],
                    'Exit_Time': ts,
                    'Exit_Price': exit_price,
                    'Score': entry_data['Score'],
                    'PnL_Points': pnl
                })
                
                print(f"🔴 TRADE CLOSED at {ts} | PnL: {pnl:.2f}")
                in_trade = False
                entry_data = {}
        
        else:
            # Check Entry Condition
            if signal_fired:
                # EXECUTE ENTRY
                in_trade = True
                entry_data = {
                    'Entry_Time': ts,
                    'Entry_Price': ltp,
                    'Score': brain.last_score
                }
                print(f"🟢 SIGNAL FIRED at {ts} | Price: {ltp} | Score: {brain.last_score:.0f}")
                # Note: We do NOT clear buffer here, as high energy might sustain
            
    print(f"--- SIMULATION ENDED. Total Trades: {len(trades)} ---")
    
    # Return DataFrame
    if trades:
        trade_df = pd.DataFrame(trades)
        print("\n=== TRADES DATAFRAME ===")
        print(trade_df)
        return trade_df
    else:
        print("No trades generated.")
        return pd.DataFrame()

if __name__ == "__main__":
    df_results = simulate_live_market("master_fut_df.csv")