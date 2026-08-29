import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timezone, time
import logging

logger = logging.getLogger(__name__)

class TimeframeGranularAnalyzer:
    """
    Analyzer for granular timeframe metrics including sessions, patterns,
    volume distribution, speed/acceleration, and wick analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        Expected columns: 'time', 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['time']):
            self.df['time'] = pd.to_datetime(self.df['time'])
        
        self.df.set_index('time', inplace=True, drop=False)
        self.df.sort_index(inplace=True)

    def analyze_all(self) -> Dict[str, Any]:
        """Run all granular analyses and return combined results."""
        try:
            return {
                "sessions": self.analyze_sessions(),
                "patterns": self.analyze_patterns(),
                "volume": self.analyze_volume(),
                "movement": self.analyze_movement(),
                "wicks": self.analyze_wicks(),
                "outlook": self.predict_outlook()
            }
        except Exception as e:
            logger.error(f"Error in granular analysis: {str(e)}")
            return {}

    def analyze_sessions(self) -> Dict[str, Any]:
        """Analyze trading sessions (Asian, London, NY)."""
        # Use current UTC time for session detection
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        
        # Define session times (UTC)
        # Asian: 00:00 - 09:00 UTC
        # London: 07:00 - 16:00 UTC
        # NY: 12:00 - 21:00 UTC
        
        # Determine current session
        session_type = 'Quiet'
        if 0 <= current_hour < 7:
            session_type = 'Asian'
        elif 7 <= current_hour < 12:
            session_type = 'London'
        elif 12 <= current_hour < 16:
            session_type = 'Overlap (London/NY)'
        elif 16 <= current_hour < 21:
            session_type = 'NY'
        elif 21 <= current_hour <= 23:
            session_type = 'Asian (Early)'
            
        # Session progress
        progress = 0
        if session_type == 'Asian':
            progress = ((current_hour - 0) / 9) * 100
        elif session_type == 'London':
            progress = ((current_hour - 7) / 9) * 100
        elif 'Overlap' in session_type:
            progress = ((current_hour - 12) / 4) * 100
        elif session_type == 'NY':
            progress = ((current_hour - 12) / 9) * 100
        elif 'Early' in session_type:
            progress = ((current_hour - 21) / 3) * 100
            
        progress = max(0, min(100, progress))

        # Calculate session statistics from DataFrame
        total_bars = len(self.df)
        bullish_bars = len(self.df[self.df['close'] > self.df['open']])
        bearish_bars = len(self.df[self.df['close'] < self.df['open']])
        
        avg_range = (self.df['high'] - self.df['low']).mean()
        avg_volume = self.df['volume'].mean()
        
        return {
            "currentSession": {
                "type": session_type,
                "progress": round(progress, 1),
                "dayOfWeek": now_utc.strftime('%A'),
                "weekOfMonth": (now_utc.day - 1) // 7 + 1,
                "month": now_utc.strftime('%B')
            },
            "totalSessions": total_bars, # Using bars as proxy for sessions for now
            "bullishSessions": bullish_bars,
            "bearishSessions": bearish_bars,
            "historicalComparison": {
                "avgRange": float(avg_range),
                "avgVolume": float(avg_volume),
                "bullishProbability": (bullish_bars / total_bars * 100) if total_bars > 0 else 0
            }
        }

    def analyze_patterns(self) -> Dict[str, Any]:
        """Detect candlestick patterns."""
        # Focus on last 3 candles for immediate patterns
        if len(self.df) < 3:
            return {"currentPatterns": [], "patternStrength": "Low", "similarHistorical": []}
            
        recent = self.df.iloc[-3:].copy()
        patterns = []
        
        # Helper for candle properties
        def get_body(row): return abs(row['close'] - row['open'])
        def get_range(row): return row['high'] - row['low']
        def is_bull(row): return row['close'] >= row['open']
        
        c1, c2, c3 = recent.iloc[0], recent.iloc[1], recent.iloc[2]
        
        # Morning Star
        if (not is_bull(c1) and 
            get_body(c2) < get_body(c1) * 0.5 and 
            is_bull(c3) and 
            c3['close'] > (c1['open'] + c1['close'])/2):
            patterns.append("Morning Star")
            
        # Evening Star
        if (is_bull(c1) and 
            get_body(c2) < get_body(c1) * 0.5 and 
            not is_bull(c3) and 
            c3['close'] < (c1['open'] + c1['close'])/2):
            patterns.append("Evening Star")
            
        # Three White Soldiers
        if (is_bull(c1) and is_bull(c2) and is_bull(c3) and
            c2['close'] > c1['close'] and c3['close'] > c2['close']):
            patterns.append("Three White Soldiers")
            
        # Doji (last candle)
        if get_range(c3) > 0 and get_body(c3) / get_range(c3) < 0.1:
            patterns.append("Doji")
            
        # Hammer / Shooting Star
        if get_range(c3) > 0:
            body_ratio = get_body(c3) / get_range(c3)
            upper_wick = c3['high'] - max(c3['open'], c3['close'])
            lower_wick = min(c3['open'], c3['close']) - c3['low']
            
            if body_ratio < 0.3:
                if lower_wick > 2 * upper_wick:
                    patterns.append("Hammer" if not is_bull(c3) else "Hanging Man") # Simplified naming
                elif upper_wick > 2 * lower_wick:
                    patterns.append("Shooting Star" if is_bull(c3) else "Inverted Hammer")

        strength = "High" if len(patterns) > 0 else "Low"
        
        return {
            "currentPatterns": patterns,
            "patternStrength": strength,
            "similarHistorical": [], # TODO: Implement historical pattern matching
            "avgOutcome": 0.0,
            "successRate": 0.0
        }

    def analyze_volume(self) -> Dict[str, Any]:
        """Analyze volume profile and trends."""
        if len(self.df) < 20:
            return {}
            
        recent_vol = self.df['volume'].iloc[-20:]
        avg_vol = recent_vol.mean()
        current_vol = self.df['volume'].iloc[-1]
        
        # Volume profile buckets
        high_vol_threshold = avg_vol * 1.5
        low_vol_threshold = avg_vol * 0.5
        
        profile = {
            "high": int((recent_vol > high_vol_threshold).sum()),
            "medium": int(((recent_vol <= high_vol_threshold) & (recent_vol >= low_vol_threshold)).sum()),
            "low": int((recent_vol < low_vol_threshold).sum()),
            "avgVolume": float(avg_vol)
        }
        
        # Trend
        vol_slope = np.polyfit(range(len(recent_vol)), recent_vol.values, 1)[0]
        trend = "Increasing" if vol_slope > 0 else "Decreasing"
        if abs(vol_slope) < avg_vol * 0.01: trend = "Mixed"
        
        return {
            "profile": profile,
            "trend": trend,
            "currentVsAvg": float(current_vol / avg_vol) if avg_vol > 0 else 1.0,
            "distribution": recent_vol.tolist()
        }

    def analyze_movement(self) -> Dict[str, Any]:
        """Analyze speed, acceleration, and momentum."""
        closes = self.df['close']
        
        # Speed: % change per bar
        speed = closes.pct_change() * 100
        current_speed = speed.iloc[-1]
        
        # Acceleration: Change in speed
        acceleration = speed.diff()
        current_accel = acceleration.iloc[-1]
        
        # Momentum
        momentum = "Steady"
        if current_accel > 0.01: momentum = "Accelerating"
        elif current_accel < -0.01: momentum = "Decelerating"
        
        # Historical context
        avg_speed = speed.abs().mean()
        recent_avg_speed = speed.iloc[-10:].abs().mean()
        
        return {
            "currentSpeed": float(current_speed),
            "currentAcceleration": float(current_accel),
            "momentum": momentum,
            "avgSpeed": float(avg_speed),
            "recentAvgSpeed": float(recent_avg_speed),
            "speedRatio": float(recent_avg_speed / avg_speed) if avg_speed > 0 else 1.0,
            "barMetrics": {
                "recentBars": {
                    "bullishCount": int((speed.iloc[-10:] > 0).sum()),
                    "bearishCount": int((speed.iloc[-10:] < 0).sum())
                }
            }
        }

    def analyze_wicks(self) -> Dict[str, Any]:
        """Analyze rejection wicks and liquidity zones."""
        recent = self.df.iloc[-20:].copy()
        wicks = []
        
        for i, (idx, row) in enumerate(recent.iterrows()):
            body_top = max(row['open'], row['close'])
            body_bottom = min(row['open'], row['close'])
            range_len = row['high'] - row['low']
            
            if range_len == 0: continue
            
            upper_wick = row['high'] - body_top
            lower_wick = body_bottom - row['low']
            
            # Significant wick threshold (e.g., > 40% of range)
            if upper_wick / range_len > 0.4:
                wicks.append({
                    "barIndex": i,
                    "type": "upper",
                    "strength": float(upper_wick / range_len * 100),
                    "priceLevel": float(row['high'])
                })
            
            if lower_wick / range_len > 0.4:
                wicks.append({
                    "barIndex": i,
                    "type": "lower",
                    "strength": float(lower_wick / range_len * 100),
                    "priceLevel": float(row['low'])
                })
                
        return {
            "rejectionWicks": wicks,
            "stopHuntLevels": [], # TODO: Implement stop hunt logic
            "liquidityZones": []  # TODO: Implement liquidity zone logic
        }

    def predict_outlook(self) -> Dict[str, Any]:
        """Predict session outlook based on current regime."""
        # Simplified outlook logic
        # In a real scenario, this would use the ML models or statistical probabilities
        
        movement = self.analyze_movement()
        vol = self.analyze_volume()
        
        regime = "Ranging"
        confidence = 60
        
        if movement['speedRatio'] > 1.5 and vol['trend'] == 'Increasing':
            regime = "Trending"
            confidence = 80
        elif movement['speedRatio'] > 2.0:
            regime = "Volatile"
            confidence = 75
            
        return {
            "candlesRemaining": 0, # Needs timeframe context to calculate
            "regimePredictions": [{
                "time": "Next 4 Hours",
                "regime": regime,
                "confidence": confidence,
                "reasoning": f"Speed ratio {movement['speedRatio']:.2f}x avg, Volume {vol['trend']}"
            }],
            "keyLevels": {
                "support": [],
                "resistance": [],
                "pivots": []
            }
        }
