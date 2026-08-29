"""
Optimized Signal Generator with Smart Chunking and Progress Aggregation

Key Features:
- Index-based chunking (contiguous time-series blocks)
- Lookback buffer overlap for context
- Aggregated progress reporting from all workers
- Memory-optimized column selection
- Data integrity validation after merge
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import Pool, Manager
from app.core.analysis.support_resistance import (
    detect_snr_levels_sequential,
    create_clustered_zones_sequential,
    extract_snr_features,
)

logger = logging.getLogger(__name__)

# ============================================================================
# REQUIRED COLUMNS FOR SIGNAL GENERATION (Memory Optimization)
# Only these columns are needed during iteration - reduces memory by 50x
# ============================================================================
REQUIRED_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


@dataclass
class ChunkConfig:
    """Configuration for a single chunk of data to process."""
    chunk_id: int
    data: pd.DataFrame  # Includes lookback buffer
    process_from_idx: int  # Where to start processing (after buffer)
    global_start_idx: int  # Original index in full DataFrame
    global_end_idx: int  # Original index in full DataFrame
    total_rows_to_process: int  # How many rows this chunk will process


@dataclass
class ChunkResult:
    """Results from processing a single chunk."""
    chunk_id: int
    signals: List[Dict]
    zones: List[Tuple]
    features: Dict[int, Dict]  # {global_index: {feature_name: value}}
    ml_dataset: List[Dict]
    signal_counts: Dict[str, int]
    rows_processed: int


@dataclass
class ProgressUpdate:
    """Unified progress update for frontend (matches AnalysisProgress.tsx expectations)."""
    progress: int  # 0-100
    message: str  # Main status message
    message2: str  # Detail message
    current_index: int
    signal_counts: Dict[str, int]
    signals_found: int
    status: str  # 'processing' | 'complete' | 'error'


def smart_chunk_dataframe(
    df: pd.DataFrame,
    n_workers: int,
    lookback_period: int,
    start_index: int,
    confirmation_period: int,
    lookforward_period: int = 50
) -> List[ChunkConfig]:
    """
    Split DataFrame into overlapping chunks for parallel SNR processing.
    
    Each chunk gets sufficient lookback buffer (lookback_period) and 
    lookahead buffer (max(confirmation_period, lookforward_period)) 
    to avoid boundary discovery gaps and movement analysis discrepancies.
    
    Args:
        df: Full DataFrame
        n_workers: Number of parallel workers
        lookback_period: How many rows before each chunk to include as context
        start_index: Where signal generation starts (after initial lookback)
        confirmation_period: Future candles needed for confirmation
        lookforward_period: Future candles needed for movement analysis
        
    Returns:
        List of ChunkConfig objects, one per worker
    """
    total_rows = len(df)
    end_index = total_rows - confirmation_period
    processable_rows = end_index - start_index
    
    if processable_rows <= 0:
        return []

    # Calculate base chunk size (approximate)
    chunk_size = processable_rows // n_workers
    if chunk_size == 0:
        chunk_size = processable_rows
        n_workers = 1

    chunks = []
    
    # Total lookahead needed for signal discovery AND movement analysis
    lookahead_buffer = max(confirmation_period, lookforward_period)
    
    for i in range(n_workers):
        # Calculate processing range for this chunk
        process_start = start_index + (i * chunk_size)
        process_end = process_start + chunk_size if i < n_workers - 1 else end_index

        # Add lookback buffer (but don't process these rows)
        buffer_start = max(0, process_start - lookback_period)
        
        # Extract data slice (buffer + processing range + lookahead)
        chunk_data = df.iloc[buffer_start : process_end + lookahead_buffer].copy()
        
        # Calculate where processing starts within this chunk
        process_from_idx = process_start - buffer_start
        
        chunks.append(ChunkConfig(
            chunk_id=i,
            data=chunk_data,
            process_from_idx=process_from_idx,
            global_start_idx=process_start,
            global_end_idx=process_end,
            total_rows_to_process=process_end - process_start
        ))
        
        logger.info(f"Chunk {i}: buffer[{buffer_start}:{process_start}] + process[{process_start}:{process_end}] + lookahead[{process_end}:{process_end + lookahead_buffer}] = {len(chunk_data)} rows total")

    return chunks


def _process_chunk_worker(
    chunk_config: ChunkConfig,
    lookback_period: int,
    confirmation_period: int,
    n_clusters: int,
    zone_width: float,
    min_distance_pct: float,
    lookforward_period: int,
    progress_queue: Any = None  # Multiprocessing Queue for progress updates
) -> ChunkResult:
    """
    Worker function to process a single chunk of data.
    
    This runs in a separate process and generates signals for its assigned range.
    
    Memory Optimization: Only loads OHLCV columns during iteration (50x reduction).
    
    Args:
        chunk_config: Configuration for this chunk
        ... (other signal generation parameters)
        progress_queue: Optional queue to send progress updates
        
    Returns:
        ChunkResult with signals, zones, features, and counts
    """
    chunk_id = chunk_config.chunk_id
    df_chunk = chunk_config.data
    process_from = chunk_config.process_from_idx
    global_start = chunk_config.global_start_idx
    
    # OPTIMIZATION: Only keep required columns for iteration
    # This reduces memory by ~50x if you have 250 columns (astronomical features)
    df_slim = df_chunk[REQUIRED_OHLCV_COLS].copy()
    
    # Initialize results
    signals = []
    zones_seen = set()  # Track unique zones
    all_zones = []
    features_dict = {}
    ml_dataset = []
    signal_counts = {
        'bounce_support': 0,
        'bounce_resistance': 0,
        'breakout_support': 0,
        'breakout_resistance': 0
    }
    
    # Calculate indices to process (within chunk)
    chunk_end = len(df_slim) - confirmation_period
    rows_to_process = chunk_end - process_from
    
    logger.info(f"Worker {chunk_id}: Processing rows {process_from} to {chunk_end} ({rows_to_process} rows)")
    
    # Process each index in this chunk's range
    for local_idx in range(process_from, chunk_end):
        global_idx = global_start + (local_idx - process_from)
        
        # Send progress update every 10 rows
        if progress_queue and (local_idx - process_from) % 10 == 0:
            progress_pct = int(((local_idx - process_from) / rows_to_process) * 100)
            progress_queue.put({
                'chunk_id': chunk_id,
                'progress': progress_pct,
                'current_index': global_idx,
                'signal_counts': signal_counts.copy(),
                'rows_processed': local_idx - process_from
            })
        
        try:
            # Detect S&R levels (using full chunk data up to current point)
            current_levels = detect_snr_levels_sequential(
                df_chunk, local_idx, lookback_period, min_distance_pct
            )
            
            if not current_levels:
                continue
            
            # Create zones
            price_data_slice = df_chunk.iloc[max(0, local_idx - lookback_period):local_idx + 1]
            current_zones = create_clustered_zones_sequential(
                current_levels,
                price_data_slice,
                n_clusters=n_clusters,
                zone_width=zone_width
            )
            
            # Store unique zones
            for zone in current_zones:
                zone_id = zone[0]
                if zone_id not in zones_seen:
                    zones_seen.add(zone_id)
                    all_zones.append(zone)
            
            # Extract SNR features
            current_candle = df_slim.iloc[local_idx]
            current_price = current_candle['Close']
            snr_feats = extract_snr_features(current_price, current_levels, current_zones)
            features_dict[global_idx] = snr_feats
            
            if not current_zones:
                continue
            
            # Check for signals at each zone
            for zone_data in current_zones:
                zone_id, zone_price, zone_levels, volume_data = zone_data
                distance_pct = abs(current_price - zone_price) / zone_price
                
                if distance_pct <= zone_width:
                    future_candles = df_slim.iloc[local_idx + 1:local_idx + 1 + confirmation_period]
                    if len(future_candles) < confirmation_period:
                        continue
                    
                    # Determine zone type
                    support_count = sum(1 for l in zone_levels if l[2] == "support")
                    res_count = sum(1 for l in zone_levels if l[2] == "resistance")
                    z_type = "support" if support_count > res_count else "resistance"
                    
                    signal_data = None
                    
                    # Check for bounce/breakout patterns
                    if z_type == "support":
                        bounced = all(c['Low'] >= zone_price * 0.995 for _, c in future_candles.iterrows())
                        if bounced and future_candles['Close'].iloc[-1] > current_price:
                            signal_data = {"type": "bounce_support", "price": zone_price}
                        elif current_price < zone_price and all(c['Close'] < zone_price * 1.005 for _, c in future_candles.iterrows()):
                            signal_data = {"type": "breakout_support", "price": zone_price}
                    else:  # resistance
                        bounced = all(c['High'] <= zone_price * 1.005 for _, c in future_candles.iterrows())
                        if bounced and future_candles['Close'].iloc[-1] < current_price:
                            signal_data = {"type": "bounce_resistance", "price": zone_price}
                        elif current_price > zone_price and all(c['Close'] > zone_price * 0.995 for _, c in future_candles.iterrows()):
                            signal_data = {"type": "breakout_resistance", "price": zone_price}
                    
                    if signal_data:
                        # Enrich signal with metadata
                        signal_data.update({
                            'index': global_idx,
                            'current_price': current_price,
                            'level_type': z_type,
                            'confirmation_period': confirmation_period,
                            'volume': current_candle['Volume'],
                            'zonal_total_volume': volume_data['total_volume'],
                            'zonal_net_volume': volume_data['net_volume']
                        })
                        
                        signals.append(signal_data)
                        signal_counts[signal_data['type']] += 1
                        
                        # Create ML record
                        ml_dataset.append({
                            'sequence': df_slim.iloc[max(0, local_idx - lookback_period):local_idx + 1].to_dict(orient='records'),
                            'targets': signal_data,
                            'metadata': {'index': global_idx, 'type': signal_data['type']}
                        })
        
        except Exception as e:
            logger.warning(f"Worker {chunk_id}: Error at index {global_idx}: {e}")
            continue
    
    # Final progress update
    if progress_queue:
        progress_queue.put({
            'chunk_id': chunk_id,
            'progress': 100,
            'current_index': global_start + rows_to_process,
            'signal_counts': signal_counts.copy(),
            'rows_processed': rows_to_process,
            'status': 'complete'
        })
    
    logger.info(f"Worker {chunk_id}: Complete. Found {len(signals)} signals, {len(all_zones)} zones")
    
    return ChunkResult(
        chunk_id=chunk_id,
        signals=signals,
        zones=all_zones,
        features=features_dict,
        ml_dataset=ml_dataset,
        signal_counts=signal_counts,
        rows_processed=rows_to_process
    )


def aggregate_progress_updates(
    chunk_updates: List[Dict],
    total_rows: int,
    total_chunks: int
) -> ProgressUpdate:
    """
    Aggregate progress from multiple chunks into a single frontend-ready update.
    
    This matches the format expected by AnalysisProgress.tsx and SNRAnalysisStepPanel.tsx.
    
    Args:
        chunk_updates: List of progress dicts from each chunk
        total_rows: Total rows being processed across all chunks
        total_chunks: Total number of chunks
        
    Returns:
        ProgressUpdate object ready for frontend consumption
    """
    # Aggregate signal counts
    total_counts = {
        'bounce_support': 0,
        'bounce_resistance': 0,
        'breakout_support': 0,
        'breakout_resistance': 0
    }
    
    total_rows_processed = 0
    max_index = 0
    completed_chunks = 0
    
    for update in chunk_updates:
        if 'signal_counts' in update:
            for key, val in update['signal_counts'].items():
                total_counts[key] += val
        
        total_rows_processed += update.get('rows_processed', 0)
        max_index = max(max_index, update.get('current_index', 0))
        
        if update.get('status') == 'complete':
            completed_chunks += 1
    
    # Calculate overall progress
    progress_pct = int((total_rows_processed / total_rows) * 100) if total_rows > 0 else 0
    
    # Generate messages (matches frontend expectations)
    total_signals = sum(total_counts.values())
    bounces = total_counts['bounce_support'] + total_counts['bounce_resistance']
    breakouts = total_counts['breakout_support'] + total_counts['breakout_resistance']
    
    message = f"PARALLEL SCAN: {completed_chunks}/{total_chunks} workers | BOUNCES: {bounces} | BREAKOUTS: {breakouts}"
    message2 = f"Processing index {max_index} | {total_signals} signals found | {total_rows_processed}/{total_rows} rows"
    
    return ProgressUpdate(
        progress=progress_pct,
        message=message,
        message2=message2,
        current_index=max_index,
        signal_counts=total_counts,
        signals_found=total_signals,
        status='processing' if completed_chunks < total_chunks else 'complete'
    )


def generate_signals_parallel_optimized(
    price_data: pd.DataFrame,
    confirmation_period: int = 3,
    lookback_period: int = 50,
    n_clusters: int = 5,
    zone_width: float = 0.004,
    min_distance_pct: float = 0.5,
    lookforward_period: int = 50,
    task_id: str = None,
    progress_store: Any = None,
    n_workers: Optional[int] = None
) -> Tuple[List[Dict], List[Tuple], pd.DataFrame, List[Dict]]:
    """
    Generate signals using optimized parallel processing with smart chunking.
    
    Features:
    - Index-based chunking (contiguous time-series blocks)
    - Lookback buffer overlap for context
    - Aggregated progress reporting
    - Memory-optimized column selection
    - Data integrity validation
    
    Returns:
        (signals, zones, enriched_dataframe, ml_dataset)
    """
    if n_workers is None:
        import multiprocessing
        n_workers_val = int(max(1, multiprocessing.cpu_count() - 1))
    else:
        n_workers_val = int(n_workers)
    
    # Prepare DataFrame
    df = price_data.copy()
    standard_map = {c.lower(): c.capitalize() for c in price_data.columns}
    df = df.rename(columns=standard_map)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    # Verify required columns
    missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    
    # Calculate start index
    start_index = lookback_period
    total_rows = len(df)
    
    if total_rows < start_index + confirmation_period + 1:
        raise ValueError(f"Insufficient data: need {start_index + confirmation_period + 1}, have {total_rows}")
    
    # Create chunks
    chunks = smart_chunk_dataframe(df, n_workers_val, lookback_period, start_index, confirmation_period)
    
    logger.info(f"Starting parallel processing: {len(chunks)} chunks, {total_rows} rows")
    
    # Create progress queue
    manager = Manager()
    progress_queue = manager.Queue()
    
    # Process chunks in parallel
    with Pool(n_workers_val) as pool:
        # Start async processing
        async_results = []
        for chunk in chunks:
            result = pool.apply_async(
                _process_chunk_worker,
                args=(chunk, lookback_period, confirmation_period, n_clusters, 
                      zone_width, min_distance_pct, lookforward_period, progress_queue)
            )
            async_results.append(result)
        
        # Monitor progress while workers run
        completed_workers = 0
        chunk_progress = {i: {} for i in range(len(chunks))}
        
        import time
        while completed_workers < len(chunks):
            try:
                # Use block=True with timeout instead of empty() to avoid 100% CPU spinning
                # which causes GIL contention and starves the FastAPI event loop (WebSocket timeout)
                try:
                    import queue
                    update = progress_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue
                
                chunk_id = update['chunk_id']
                chunk_progress[chunk_id] = update
                
                if update.get('status') == 'complete':
                    completed_workers += 1
                
                # Aggregate and send to frontend
                aggregated = aggregate_progress_updates(
                    list(chunk_progress.values()),
                    total_rows - start_index - confirmation_period,
                    len(chunks)
                )
                
                if progress_store and task_id:
                    progress_store.update_task(
                        task_id,
                        progress=aggregated.progress,
                        message=aggregated.message,
                        message2=aggregated.message2,
                        current_index=aggregated.current_index,
                        signal_counts=aggregated.signal_counts,
                        signals_found=aggregated.signals_found
                    )
            except Exception as e:
                logger.warning(f"Error checking progress update: {e}")
        
        # Collect results
        chunk_results = [ar.get() for ar in async_results]
    
    # Merge results
    all_signals = []
    all_zones = []
    all_features = {}
    all_ml_data = []
    final_counts = {
        'bounce_support': 0,
        'bounce_resistance': 0,
        'breakout_support': 0,
        'breakout_resistance': 0
    }
    
    for result in chunk_results:
        all_signals.extend(result.signals)
        all_zones.extend(result.zones)
        all_features.update(result.features)
        all_ml_data.extend(result.ml_dataset)
        for key, val in result.signal_counts.items():
            final_counts[key] += val
    
    # DATA INTEGRITY CHECK
    expected_features = len(df) - start_index - confirmation_period
    actual_features = len(all_features)
    
    if actual_features != expected_features:
        logger.warning(
            f"⚠️ Feature count mismatch: expected {expected_features}, got {actual_features}. "
            f"Difference: {expected_features - actual_features} rows"
        )
    else:
        logger.info(f"✅ Data integrity check passed: {actual_features} features match expected count")
    
    # Apply features to DataFrame
    for idx, feats in all_features.items():
        for k, v in feats.items():
            df.at[idx, k] = v
    
    logger.info(f"✅ Parallel processing complete: {len(all_signals)} signals, {len(all_zones)} zones")
    
    return all_signals, all_zones, df, all_ml_data
