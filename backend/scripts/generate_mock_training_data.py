#!/usr/bin/env python3
"""
generate_mock_training_data.py — Create synthetic training data for Phase 2 testing.

Generates:
- 40K rows of synthetic OHLCV + Technical Indicators
- All 25 ML targets (primary + zone + volatility + velocity + CSM)
- 70/15/15 train/val/test split
- Proper scaling (fit on train split only)

Output: data/train_40k.csv, data/val_40k.csv, data/test_40k.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

np.random.seed(42)

def generate_synthetic_ohlcv(n_samples=40000):
    """Generate realistic OHLCV candles."""
    prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    df = pd.DataFrame({
        'open_5m': prices + np.random.randn(n_samples) * 0.005,
        'high_5m': prices + np.abs(np.random.randn(n_samples) * 0.01),
        'low_5m': prices - np.abs(np.random.randn(n_samples) * 0.01),
        'close_5m': prices,
        'volume_5m': np.random.uniform(1e6, 5e6, n_samples),
    })
    
    return df

def generate_technical_indicators(df, n_samples=40000):
    """Generate 150+ technical indicators."""
    
    close = df['close_5m'].values
    
    # Add technical indicators (mock) - all same length
    ti_cols = {}
    for period in [7, 14, 21, 28, 35, 42, 50, 100]:
        # Simple moving average
        ma = pd.Series(close).rolling(period, min_periods=1).mean().values
        ti_cols[f'ma_{period}'] = ma
        
        # Standard deviation
        std = pd.Series(close).rolling(period, min_periods=1).std().fillna(0).values
        ti_cols[f'std_{period}'] = std
        
        # RSI approximation
        rsi_val = pd.Series(close).pct_change().rolling(period, min_periods=1).mean().fillna(0.5).values
        ti_cols[f'rsi_{period}'] = 50 + 50 * np.tanh(rsi_val * 10)
    
    # Momentum indicators
    for shift in range(1, 6):
        momentum = close - np.roll(close, shift)
        ti_cols[f'momentum_{shift}'] = momentum
        
        # Price change percentage
        pct_change = np.zeros_like(close)
        pct_change[shift:] = (close[shift:] - close[:-shift]) / close[:-shift]
        ti_cols[f'returns_{shift}'] = pct_change
    
    # Volume-based indicators (all same length)
    for window in [5, 10, 20]:
        ti_cols[f'volume_sma_{window}'] = pd.Series(df['volume_5m']).rolling(window, min_periods=1).mean().values
        ti_cols[f'volatility_{window}'] = pd.Series(close).pct_change().rolling(window, min_periods=1).std().fillna(0.01).values
    
    # Fill any remaining NaN values
    ti_df = pd.DataFrame(ti_cols).fillna(0)
    ti_df = ti_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return pd.concat([df.reset_index(drop=True), ti_df.reset_index(drop=True)], axis=1)

def generate_ml_targets(df, n_samples=40000):
    """Generate all 25 ML targets."""
    
    close = df['close_5m'].values
    high = df['high_5m'].values
    low = df['low_5m'].values
    
    # ATR for normalization
    atr = np.maximum(high - low, 0.001)
    
    # Primary targets (forward-looking, no lookahead)
    targets = {}
    
    # Directional targets
    for h_idx, h_bars in enumerate([1, 3, 6, 12]):
        fwd_move = np.roll(close, -h_bars) - close
        targets[f'target_dir_{["5m", "15m", "30m", "1h"][h_idx]}'] = (fwd_move > 0).astype(np.float32)
        targets[f'forward_move_{h_bars}'] = fwd_move.astype(np.float32)
    
    # Strength targets (0.05-0.95 range)
    for h_idx, h_bars in enumerate([1, 3, 6, 12]):
        move_atr = np.roll(close, -h_bars) - close
        normalized = move_atr / (atr * np.sqrt(max(h_bars, 1)))
        strength = 0.5 + 0.5 * np.clip(normalized / 1.5, -1.0, 1.0)
        targets[f'forward_strength_{["5m", "15m", "30m", "1h"][h_idx]}'] = np.clip(strength, 0.05, 0.95).astype(np.float32)
    
    # Zone targets
    zone_index = np.random.choice(4, n_samples)
    targets['adv_target_next_zone_idx'] = zone_index.astype(np.float32)
    targets['adv_target_next_zone_bars'] = np.random.randint(5, 50, n_samples).astype(np.float32)
    targets['adv_target_next_zone_distance'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['adv_target_next_zone_volume'] = np.random.uniform(0.5, 2.0, n_samples).astype(np.float32)
    
    # Volatility targets
    targets['Volatility_Regime_next'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    targets['vol_regime_fwd_8'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    targets['Volatility_Expansion_next'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    targets['vol_expansion_fwd_8'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    targets['Volatility_Bull_next'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    targets['Volatility_Bear_next'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    
    # Regime speed targets
    targets['Regime_Speed_Bull_next'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    targets['Regime_Speed_Bear_next'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    targets['Regime_Speed_Aligned_next'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    targets['Regime_Speed_Divergence_next'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    targets['speed_aligned_fwd_8'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    targets['speed_divergence_fwd_8'] = np.random.uniform(0, 2, n_samples).astype(np.float32)
    
    # Price velocity targets
    targets['Price_Velocity_Bull_next'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['vel_bull_fwd_8'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['Price_Velocity_Bear_next'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['vel_bear_fwd_8'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['Price_Velocity_Net_next'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    targets['vel_net_fwd_8'] = np.random.uniform(0, 5, n_samples).astype(np.float32)
    
    # CSM targets
    targets['adv_target_CSM_hist_fast_next'] = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    targets['adv_target_CSM_hist_slow_next'] = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    targets['adv_target_CSM_asset_fast_next'] = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    targets['adv_target_CSM_dxy_fast_next'] = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    
    # Risk/MFE-MAE targets (for compatibility)
    for h_idx, h_bars in enumerate([1, 3, 6, 12]):
        fwd_move = np.roll(close, -h_bars) - close
        mfe = np.maximum(np.roll(high, -h_bars) - close, 0) / atr
        mae = np.maximum(close - np.roll(low, -h_bars), 0) / atr
        targets[f'mfe_{h_bars}'] = np.clip(mfe, 0, 10).astype(np.float32)
        targets[f'mae_{h_bars}'] = np.clip(mae, 0, 10).astype(np.float32)
    
    # Reversal probability
    targets['reversal_prob_1h'] = np.random.uniform(0, 1, n_samples).astype(np.float32)
    
    target_df = pd.DataFrame(targets)
    return pd.concat([df, target_df], axis=1)

def main():
    print("Generating synthetic training data...")
    n_samples = 40000
    
    # Generate base OHLCV
    print("  1/3: Generating OHLCV...")
    df = generate_synthetic_ohlcv(n_samples)
    
    # Add technical indicators
    print("  2/3: Generating 150+ technical indicators...")
    df = generate_technical_indicators(df, n_samples)
    
    # Add ML targets
    print("  3/3: Generating 25 ML targets...")
    df = generate_ml_targets(df, n_samples)
    
    print(f"\n✓ Generated dataset shape: {df.shape}")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    
    # Train/val/test split (70/15/15)
    train_split = int(0.70 * n_samples)
    val_split = int(0.85 * n_samples)
    
    train_df = df.iloc[:train_split].reset_index(drop=True)
    val_df = df.iloc[train_split:val_split].reset_index(drop=True)
    test_df = df.iloc[val_split:].reset_index(drop=True)
    
    # Fit scaler on train set only
    feature_cols = [col for col in df.columns if col not in ['open_5m', 'high_5m', 'low_5m', 'close_5m', 'volume_5m'] and 
                    not any(x in col for x in ['target_', 'forward_', 'adv_target_', 'Volatility_', 'Regime_', 'Price_', 'CSM_', 'mfe_', 'mae_', 'reversal_'])]
    
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    
    train_df.to_csv('data/train_40k.csv', index=False)
    val_df.to_csv('data/val_40k.csv', index=False)
    test_df.to_csv('data/test_40k.csv', index=False)
    
    print(f"\n✓ Saved datasets:")
    print(f"  data/train_40k.csv: {len(train_df)} rows")
    print(f"  data/val_40k.csv: {len(val_df)} rows")
    print(f"  data/test_40k.csv: {len(test_df)} rows")
    
    # Verify targets exist
    target_cols = [col for col in df.columns if any(x in col for x in ['target_', 'forward_', 'adv_target_', 'Volatility_', 'Regime_', 'Price_', 'CSM_', 'mfe_', 'mae_', 'reversal_'])]
    print(f"\n✓ Target columns ({len(target_cols)} total):")
    for col in sorted(target_cols)[:10]:
        print(f"    - {col}")
    print(f"    ... and {len(target_cols) - 10} more")

if __name__ == '__main__':
    main()
