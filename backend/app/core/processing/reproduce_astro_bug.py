
import pandas as pd
import numpy as np
import logging
import sys

# Mocking some parts of the system
class AnalysisType:
    ASTRONOMICAL = "astronomical"
    SNR = "snr"

def simulate_astro_bug():
    # 1. Create original_df (as if coming from SNR)
    n_rows = 100
    data = {
        'Time': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
        'Open': np.random.randn(n_rows),
        'High': np.random.randn(n_rows),
        'Low': np.random.randn(n_rows),
        'Close': np.random.randn(n_rows),
        'Volume': np.random.randn(n_rows),
        'Existing_Feature': np.random.randn(n_rows)
    }
    original_df = pd.DataFrame(data)
    
    # 2. Simulate ParallelChunkingStrategy._prepare_dataframe
    df_prepared = original_df.copy()
    df_prepared = df_prepared.reset_index(drop=True)
    
    # 3. Simulate Worker result
    # Astro worker returns 'Date' and new features, but NO 'Time', 'Open' etc.
    # IMPORTANT: The columns might be in a different order than expected!
    worker_data = {
        'Astro_F1': np.ones(n_rows), # All 1s
        'Astro_F2': np.zeros(n_rows),
        'Date': original_df['Time'].tolist()
    }
    result_df = pd.DataFrame(worker_data)
    
    # 4. Simulate _ensure_result_completeness
    print(f"Original Columns: {original_df.columns.tolist()}")
    print(f"Result Columns (before enrichment): {result_df.columns.tolist()}")
    
    # Deduplication
    result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')].copy()
    original_df = original_df.loc[:, ~original_df.columns.duplicated(keep='first')].copy()
    
    # Rename
    core_map = {'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    result_df = result_df.rename(columns={c: core_map.get(c.lower(), c) for c in result_df.columns})
    original_df = original_df.rename(columns={c: core_map.get(c.lower(), c) for c in original_df.columns})
    
    # Alignment (skip length check as they are equal)
    if len(result_df) == len(original_df):
        # Even if equal, the code might still run if types differ
        # Simulate the 'else' block which uses .values
        new_result = pd.DataFrame(index=original_df.index, columns=result_df.columns, dtype='object')
        tail_rows = result_df.iloc[-len(result_df):].values
        new_result.iloc[-len(result_df):, :] = tail_rows
        result_df = new_result
    
    # Merge missing columns
    missing_cols = [c for c in original_df.columns if c not in result_df.columns]
    for col in missing_cols:
        result_df[col] = original_df[col].values # Simplified for test
        
    print(f"Missing columns added: {missing_cols}")
    
    # Reorder
    original_col_order = [c for c in original_df.columns if c in result_df.columns]
    new_analysis_cols = [c for c in result_df.columns if c not in original_df.columns]
    result_df = result_df[original_col_order + new_analysis_cols]
    
    print(f"Final Columns: {result_df.columns.tolist()}")
    
    # Check if 'Time' is still 'Time'
    if not pd.api.types.is_datetime64_any_dtype(result_df['Time']):
        print(f"ERROR: Time column type is {result_df['Time'].dtype}")
    else:
        print("Time column type is correct (datetime)")
        
    print(f"First 5 rows of Time:\n{result_df['Time'].head()}")
    print(f"First 5 rows of Astro_F1 (should be 1s):\n{result_df['Astro_F1'].head()}")

if __name__ == "__main__":
    simulate_astro_bug()
