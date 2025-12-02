import numpy as np
from collections import deque
import time

class KineticBrain:

    def __init__(self, threshold=37500, hold_seconds=900):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.tick_buffer = deque(maxlen=50) 
        self.in_trade = False
        self.entry_time = 0
        self.last_score = 0

    def process_tick(self, ltp, cumulative_volume):
        current_time = time.time()
        self.tick_buffer.append([ltp, cumulative_volume])
        #wait 50 tick
        if len(self.tick_buffer) < 50:
            return 0
        if self.in_trade:
            time_passed = current_time - self.entry_time
            if time_passed >= self.hold_seconds:
                self.in_trade = False
                self.entry_time = 0
                return -1 # exit
            else:
                return 0 # hold
        score = self._calculate_score()
        self.last_score = score
        if score > self.threshold:
            self.in_trade = True
            self.entry_time = current_time
            return 1 # enter long straddle
        return 0 # wait

    def _calculate_score(self):
        data = np.array(self.tick_buffer)
        prices = data[:, 0]
        vols = data[:, 1]
        vol_diff = np.diff(vols)
        trade_vol = np.where(vol_diff > 0, vol_diff, 0)
        displacement = abs(prices[-1] - prices[0])
        total_vol = np.sum(trade_vol)
        score = total_vol / (displacement + 0.05)
        
        return score