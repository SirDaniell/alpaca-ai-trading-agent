import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def classify_ticks(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify ticks as buy or sell using the tick rule (Lee-Ready algorithm).
    If price goes up, it is a buy. If price goes down, it is a sell.
    If unchanged, it inherits the previous direction.
    
    ticks_df must have a 'price' or 'last' column, and 'volume' or 'tick_volume'.
    """
    df = ticks_df.copy()
    if 'price' not in df.columns:
        if 'last' in df.columns:
            df['price'] = df['last']
        elif 'bid' in df.columns:
            # Fallback to mid price if only bid/ask exist
            df['price'] = (df['bid'] + df.get('ask', df['bid'])) / 2.0
        else:
            raise ValueError("Ticks DataFrame must contain 'price', 'last', or 'bid' column.")
            
    if 'volume' not in df.columns:
        df['volume'] = df.get('tick_volume', 1.0)
        
    df = df.sort_values('time')
    
    # Tick rule direction inference
    price_diff = df['price'].diff()
    
    # 1 for buy, -1 for sell, 0 for unchanged
    direction = np.sign(price_diff)
    
    # Fill 0 (unchanged) with the last non-zero direction
    direction = direction.replace(0, np.nan).ffill().fillna(1.0)
    
    df['side'] = direction.map({1.0: 'buy', -1.0: 'sell'})
    df['buy_vol'] = np.where(df['side'] == 'buy', df['volume'], 0.0)
    df['sell_vol'] = np.where(df['side'] == 'sell', df['volume'], 0.0)
    
    return df

def calculate_value_area(price_volumes: Dict[float, Dict[str, float]], total_volume: float, value_area_pct: float = 0.70) -> Tuple[float, float, float]:
    """
    Calculate Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    
    price_volumes: Dict mapping price -> {'total': val, 'buy': val, 'sell': val}
    """
    if not price_volumes:
        return 0.0, 0.0, 0.0
        
    # Sort prices
    sorted_prices = sorted(price_volumes.keys())
    
    # Find POC (price with max volume)
    poc_price = max(price_volumes.keys(), key=lambda p: price_volumes[p]['total'])
    
    if total_volume <= 0:
        return poc_price, poc_price, poc_price
        
    target_vol = total_volume * value_area_pct
    current_vol = price_volumes[poc_price]['total']
    
    # Indexes for expansion
    poc_idx = sorted_prices.index(poc_price)
    upper_idx = poc_idx
    lower_idx = poc_idx
    
    # Expand until we contain 70% of volume
    while current_vol < target_vol:
        # Check if we can expand in both directions
        has_upper = upper_idx + 1 < len(sorted_prices)
        has_lower = lower_idx - 1 >= 0
        
        if not has_upper and not has_lower:
            break
            
        vol_upper = 0.0
        if has_upper:
            # Look at next two levels up
            vol_upper = sum(price_volumes[sorted_prices[i]]['total'] for i in range(upper_idx + 1, min(upper_idx + 3, len(sorted_prices))))
            
        vol_lower = 0.0
        if has_lower:
            # Look at next two levels down
            vol_lower = sum(price_volumes[sorted_prices[i]]['total'] for i in range(max(0, lower_idx - 2), lower_idx))
            
        if vol_upper >= vol_lower and has_upper:
            # Expand upper
            next_upper = min(upper_idx + 2, len(sorted_prices) - 1)
            for i in range(upper_idx + 1, next_upper + 1):
                current_vol += price_volumes[sorted_prices[i]]['total']
            upper_idx = next_upper
        elif has_lower:
            # Expand lower
            next_lower = max(0, lower_idx - 2)
            for i in range(next_lower, lower_idx):
                current_vol += price_volumes[sorted_prices[i]]['total']
            lower_idx = next_lower
        elif has_upper:
            # Upper expansion only
            upper_idx += 1
            current_vol += price_volumes[sorted_prices[upper_idx]]['total']
            
    vah = sorted_prices[upper_idx]
    val = sorted_prices[lower_idx]
    
    return poc_price, vah, val

def calculate_imbalances(price_volumes: Dict[float, Dict[str, float]], sorted_prices: List[float], diagonal_check: bool = True) -> float:
    """
    Calculate maximum buying or selling imbalance.
    Imbalance compares buying volume at one level with selling volume at a neighboring level (usually diagonally).
    Returns the maximum imbalance ratio found in this bar.
    """
    if len(sorted_prices) < 2:
        return 1.0
        
    max_imbalance = 1.0
    
    for i in range(len(sorted_prices) - 1):
        p_low = sorted_prices[i]
        p_high = sorted_prices[i+1]
        
        # Compare diagonally: buy vol of lower price vs sell vol of higher price, or vice versa
        if diagonal_check:
            buy_val = price_volumes[p_low]['buy']
            sell_val = price_volumes[p_high]['sell']
        else:
            buy_val = price_volumes[p_high]['buy']
            sell_val = price_volumes[p_high]['sell']
            
        # Avoid division by zero
        if sell_val > 0 and buy_val > 0:
            ratio1 = buy_val / sell_val
            ratio2 = sell_val / buy_val
            max_imbalance = max(max_imbalance, ratio1, ratio2)
        elif sell_val == 0 and buy_val > 0:
            max_imbalance = max(max_imbalance, buy_val) # arbitrary high ratio
        elif buy_val == 0 and sell_val > 0:
            max_imbalance = max(max_imbalance, sell_val)
            
    return float(max_imbalance)

def build_footprint_table(bars_df: pd.DataFrame, ticks_df: pd.DataFrame, timeframe_str: str) -> pd.DataFrame:
    """
    Aggregates tick data into footprint metrics matching the index and time bounds of bars_df.
    Returns a DataFrame with columns:
    - fp_poc
    - fp_vah
    - fp_val
    - fp_delta
    - fp_imbalance_max
    - fp_high_vol_rejection
    - fp_data_available
    """
    # Ensure correct index and datetime formatting
    bars = bars_df.copy()
    if 'time' not in bars.columns:
        bars['time'] = bars.index
        
    ticks = classify_ticks(ticks_df)
    
    # We group ticks by the bar timestamps they fall into.
    # To do this safely, we construct bins from bar times.
    bar_times = sorted(bars['time'].tolist())
    if len(bar_times) < 2:
        # Create empty footprint df
        return pd.DataFrame(index=bars.index, data={
            'fp_poc': np.nan, 'fp_vah': np.nan, 'fp_val': np.nan,
            'fp_delta': 0.0, 'fp_imbalance_max': 1.0,
            'fp_high_vol_rejection': 0.0, 'fp_data_available': 0.0
        })
        
    # Estimate bar duration to bin the last bar
    durations = pd.Series(bar_times).diff().dropna()
    median_duration = durations.median()
    if pd.isna(median_duration):
        median_duration = pd.Timedelta(minutes=5) # default fallback
        
    bins = bar_times + [bar_times[-1] + median_duration]
    
    # Assign ticks to bar bins
    ticks['bar_time'] = pd.cut(ticks['time'], bins=bins, labels=bar_times, right=False)
    
    # Group ticks by bar_time
    grouped = ticks.groupby('bar_time', observed=False)
    
    fp_results = []
    
    for bar_time in bar_times:
        if bar_time not in grouped.groups:
            # No tick data for this bar
            fp_results.append({
                'time': bar_time,
                'fp_poc': np.nan,
                'fp_vah': np.nan,
                'fp_val': np.nan,
                'fp_delta': 0.0,
                'fp_imbalance_max': 1.0,
                'fp_high_vol_rejection': 0.0,
                'fp_data_available': 0.0
            })
            continue
            
        group_df = grouped.get_group(bar_time)
        if group_df.empty:
            fp_results.append({
                'time': bar_time,
                'fp_poc': np.nan,
                'fp_vah': np.nan,
                'fp_val': np.nan,
                'fp_delta': 0.0,
                'fp_imbalance_max': 1.0,
                'fp_high_vol_rejection': 0.0,
                'fp_data_available': 0.0
            })
            continue
            
        # Classify volumes at price levels. We round prices slightly to group ticks.
        # For Forex, rounding to 5 decimal places or less is common.
        # Let's dynamically find price step/tick size.
        prices = group_df['price'].unique()
        if len(prices) > 1:
            diffs = np.diff(np.sort(prices))
            tick_size = float(np.min(diffs[diffs > 0])) if len(diffs[diffs > 0]) > 0 else 0.00001
        else:
            tick_size = 0.00001
            
        # Standardize rounding
        round_decimals = int(np.ceil(-np.log10(tick_size))) if tick_size > 0 else 5
        group_df = group_df.copy()
        group_df['price_level'] = group_df['price'].round(round_decimals)
        
        # Build price-volume map
        pv_group = group_df.groupby('price_level')
        price_volumes = {}
        total_vol = 0.0
        total_buy_vol = 0.0
        total_sell_vol = 0.0
        
        for price_level, pv_df in pv_group:
            v_tot = pv_df['volume'].sum()
            v_buy = pv_df['buy_vol'].sum()
            v_sell = pv_df['sell_vol'].sum()
            
            price_volumes[float(price_level)] = {
                'total': float(v_tot),
                'buy': float(v_buy),
                'sell': float(v_sell)
            }
            total_vol += v_tot
            total_buy_vol += v_buy
            total_sell_vol += v_sell
            
        poc, vah, val = calculate_value_area(price_volumes, total_vol)
        delta = total_buy_vol - total_sell_vol
        sorted_prices = sorted(price_volumes.keys())
        imbalance_max = calculate_imbalances(price_volumes, sorted_prices)
        
        # High volume rejection: high volume near High/Low of the bar, but price rejected
        # Find if POC is in the top 15% or bottom 15% of the bar's price range
        bar_high = group_df['price'].max()
        bar_low = group_df['price'].min()
        bar_range = bar_high - bar_low
        
        high_vol_rejection = 0.0
        if bar_range > 0:
            poc_position = (poc - bar_low) / bar_range
            if (poc_position < 0.15 or poc_position > 0.85) and total_vol > ticks['volume'].mean() * 5:
                # Strong rejection pin-bar like volume concentration
                high_vol_rejection = 1.0
                
        fp_results.append({
            'time': bar_time,
            'fp_poc': poc,
            'fp_vah': vah,
            'fp_val': val,
            'fp_delta': delta,
            'fp_imbalance_max': imbalance_max,
            'fp_high_vol_rejection': high_vol_rejection,
            'fp_data_available': 1.0
        })
        
    fp_df = pd.DataFrame(fp_results)
    fp_df = fp_df.set_index('time')
    return fp_df
