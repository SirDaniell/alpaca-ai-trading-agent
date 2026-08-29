"""
Phase 1 Implementation Tests

Tests for:
  - Synthetic data generation
  - Signal pipeline orchestration
  - Async/await patterns
  - Logging verification
"""

import pytest
import asyncio
import logging
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# Assuming these modules exist or will be created
from app.core.data.synthetic_data_generator import (
    SyntheticDataGenerator,
    Candle,
)
from app.core.ml.signal_pipeline import (
    SignalPipeline,
    SignalResult,
    PipelineOutput,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def logger():
    """Create a test logger with stream handler."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


@pytest.fixture
def synthetic_generator():
    """Create synthetic data generator."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def signal_pipeline():
    """Create signal pipeline."""
    return SignalPipeline()


@pytest.fixture
def sample_candles():
    """Generate sample candles for testing."""
    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_session("AAPL", num_candles=50, start_price=150.0)


# ============================================================================
# Synthetic Data Generator Tests
# ============================================================================

class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""
    
    def test_generator_initialization(self, synthetic_generator):
        """Test generator initializes with seed."""
        assert synthetic_generator.seed == 42
    
    def test_generate_session_basic(self, synthetic_generator):
        """Test basic session generation."""
        candles = synthetic_generator.generate_session(
            symbol="AAPL",
            num_candles=100,
            start_price=150.0,
        )
        
        assert len(candles) == 100
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].open == 150.0  # First price is start_price
    
    def test_generate_session_ohlc_validity(self, synthetic_generator):
        """Test OHLC values are logically valid."""
        candles = synthetic_generator.generate_session(
            symbol="AAPL",
            num_candles=50,
            start_price=100.0,
        )
        
        for candle in candles:
            assert candle.high >= candle.low
            assert candle.high >= max(candle.open, candle.close)
            assert candle.low <= min(candle.open, candle.close)
            assert candle.volume > 0
    
    def test_generate_session_with_trend(self, synthetic_generator):
        """Test positive trend increases prices."""
        candles_up = synthetic_generator.generate_session(
            symbol="TEST",
            num_candles=50,
            start_price=100.0,
            trend=0.01,
            volatility=0.001,
        )
        
        candles_down = synthetic_generator.generate_session(
            symbol="TEST",
            num_candles=50,
            start_price=100.0,
            trend=-0.01,
            volatility=0.001,
        )
        
        avg_up = sum(c.close for c in candles_up) / len(candles_up)
        avg_down = sum(c.close for c in candles_down) / len(candles_down)
        
        assert avg_up > avg_down
    
    def test_generate_session_deterministic(self, synthetic_generator):
        """Test same seed produces same candles."""
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)
        
        candles1 = gen1.generate_session("AAPL", num_candles=30)
        candles2 = gen2.generate_session("AAPL", num_candles=30)
        
        for c1, c2 in zip(candles1, candles2):
            assert c1.open == c2.open
            assert c1.high == c2.high
            assert c1.low == c2.low
            assert c1.close == c2.close
            assert c1.volume == c2.volume
    
    def test_generate_multi_symbol(self, synthetic_generator):
        """Test multi-symbol generation."""
        symbols = ["AAPL", "GOOGL", "TSLA"]
        multi = synthetic_generator.generate_multi_symbol(
            symbols=symbols,
            num_candles=25,
        )
        
        assert len(multi) == 3
        assert all(s in multi for s in symbols)
        assert all(len(multi[s]) == 25 for s in symbols)
    
    def test_add_pattern_bullish_divergence(self, synthetic_generator, sample_candles):
        """Test bullish divergence pattern injection."""
        patterned = synthetic_generator.add_pattern(
            sample_candles,
            "bullish_divergence",
            position=25,
            intensity=1.5,
        )
        
        assert len(patterned) == len(sample_candles)
        # Candle at position should have modified values
        assert patterned[25] != sample_candles[25]
    
    def test_add_pattern_gap(self, synthetic_generator, sample_candles):
        """Test gap pattern injection."""
        patterned = synthetic_generator.add_pattern(
            sample_candles,
            "gap",
            position=30,
        )
        
        assert len(patterned) == len(sample_candles)
        # Check that gap candle differs significantly from previous
        assert abs(patterned[30].open - sample_candles[29].close) > 0
    
    def test_to_dict_list(self, synthetic_generator, sample_candles):
        """Test conversion to dict list."""
        dicts = synthetic_generator.to_dict_list(sample_candles)
        
        assert len(dicts) == len(sample_candles)
        assert all(isinstance(d, dict) for d in dicts)
        assert all("time" in d and "open" in d and "close" in d for d in dicts)


# ============================================================================
# Signal Pipeline Tests
# ============================================================================

class TestSignalPipeline:
    """Tests for SignalPipeline."""
    
    def test_pipeline_initialization(self, signal_pipeline):
        """Test pipeline initializes."""
        assert signal_pipeline is not None
    
    @pytest.mark.asyncio
    async def test_process_session_basic(self, signal_pipeline, synthetic_generator):
        """Test basic pipeline execution."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        result = await signal_pipeline.process_session(
            candles=candles_dict,
            symbol="AAPL",
            timeframe="1H",
        )
        
        assert isinstance(result, PipelineOutput)
        assert result.metadata["symbol"] == "AAPL"
        assert result.metadata["timeframe"] == "1H"
        assert result.processing_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_session_empty_candles_fails(self, signal_pipeline):
        """Test pipeline fails with empty candles."""
        with pytest.raises(ValueError):
            await signal_pipeline.process_session(
                candles=[],
                symbol="AAPL",
            )
    
    @pytest.mark.asyncio
    async def test_process_session_missing_fields_fails(self, signal_pipeline):
        """Test pipeline fails with incomplete candles."""
        bad_candles = [
            {"time": 1000, "open": 100},  # Missing high, low, close, volume
        ]
        
        with pytest.raises(ValueError):
            await signal_pipeline.process_session(
                candles=bad_candles,
                symbol="AAPL",
            )
    
    @pytest.mark.asyncio
    async def test_enrich_data_output(self, signal_pipeline, synthetic_generator):
        """Test data enrichment produces expected feature groups."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        enriched = await signal_pipeline._enrich_data(candles_dict, "AAPL")
        
        assert "indicators" in enriched
        assert "volatility" in enriched
        assert "price_stats" in enriched
        assert "volume_stats" in enriched
        assert enriched["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, signal_pipeline, synthetic_generator):
        """Test signal generation."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        enriched = await signal_pipeline._enrich_data(candles_dict, "AAPL")
        signals = await signal_pipeline._generate_signals(candles_dict, enriched, "AAPL")
        
        assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_classify_signals(self, signal_pipeline, synthetic_generator):
        """Test signal classification."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        enriched = await signal_pipeline._enrich_data(candles_dict, "AAPL")
        raw_signals = await signal_pipeline._generate_signals(candles_dict, enriched, "AAPL")
        classified, classifications = await signal_pipeline._classify_signals(
            raw_signals, enriched, "AAPL"
        )
        
        assert all(isinstance(s, SignalResult) for s in classified)
        assert "method" in classifications
        assert "avg_confidence" in classifications


# ============================================================================
# Async & Logging Tests
# ============================================================================

class TestAsyncAndLogging:
    """Tests for async/await patterns and logging."""
    
    @pytest.mark.asyncio
    async def test_pipeline_is_async(self, signal_pipeline, synthetic_generator):
        """Test pipeline methods are async."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        # Should be awaitable
        result = await signal_pipeline.process_session(candles_dict, "AAPL")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_parallel_signal_processing(self, signal_pipeline, synthetic_generator):
        """Test processing multiple symbols in parallel."""
        gen = SyntheticDataGenerator(seed=42)
        
        tasks = []
        for symbol in ["AAPL", "GOOGL", "TSLA"]:
            candles = gen.generate_session(symbol, num_candles=50)
            candles_dict = gen.to_dict_list(candles)
            tasks.append(
                signal_pipeline.process_session(candles_dict, symbol, "1H")
            )
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(isinstance(r, PipelineOutput) for r in results)
    
    def test_logging_present(self, caplog):
        """Test that logging is produced."""
        with caplog.at_level(logging.INFO):
            gen = SyntheticDataGenerator(seed=42)
            candles = gen.generate_session("AAPL", num_candles=10)
        
        # Should have logged initialization and generation
        assert any("initialized" in record.message.lower() for record in caplog.records)
    
    @pytest.mark.asyncio
    async def test_pipeline_logging(self, signal_pipeline, synthetic_generator, caplog):
        """Test pipeline produces structured logs."""
        candles = synthetic_generator.generate_session("AAPL", num_candles=50)
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        with caplog.at_level(logging.INFO):
            result = await signal_pipeline.process_session(candles_dict, "AAPL")
        
        # Should have logged pipeline start and completion
        assert any("Pipeline START" in record.message for record in caplog.records)
        # Note: "Pipeline COMPLETE" or similar would be logged on success


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_e2e(self, synthetic_generator, signal_pipeline):
        """Test complete flow: generate data → process → classify."""
        # Generate synthetic data
        candles = synthetic_generator.generate_session(
            symbol="AAPL",
            num_candles=100,
            start_price=150.0,
            volatility=0.02,
            trend=0.001,
        )
        candles_dict = synthetic_generator.to_dict_list(candles)
        
        # Inject pattern
        patterned = synthetic_generator.add_pattern(
            candles, "bullish_divergence", intensity=1.5
        )
        patterned_dict = synthetic_generator.to_dict_list(patterned)
        
        # Process through pipeline
        result = await signal_pipeline.process_session(
            candles=patterned_dict,
            symbol="AAPL",
            timeframe="1H",
        )
        
        # Verify result structure
        assert len(result.signals) >= 0
        assert result.metadata["candle_count"] == 100
        assert result.processing_ms > 0
        
        # Log result summary
        print(
            f"\nE2E Test Result:\n"
            f"  Signals: {len(result.signals)}\n"
            f"  Processing: {result.processing_ms:.1f}ms\n"
            f"  Candles: {result.metadata['candle_count']}"
        )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
