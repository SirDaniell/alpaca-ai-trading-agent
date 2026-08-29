"""
Indicator Configuration Management Module
Provides preset configurations and parameter validation for all technical indicators
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class PresetType(str, Enum):
    """Predefined configuration presets"""

    DEFAULT = "default"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    DAY_TRADING = "day_trading"


class IndicatorPreset(BaseModel):
    """Preset configuration for an indicator"""

    name: str
    description: str
    parameters: Dict[str, Any]
    recommended_timeframes: List[str]
    use_case: str


# =====================================================================
# INDICATOR PRESETS DATABASE
# =====================================================================

INDICATOR_PRESETS: Dict[str, Dict[PresetType, IndicatorPreset]] = {
    "rsi": {
        PresetType.DEFAULT: IndicatorPreset(
            name="Standard RSI",
            description="Traditional 14-period RSI",
            parameters={"length": 14},
            recommended_timeframes=["15m", "1h", "4h", "1d"],
            use_case="General overbought/oversold detection",
        ),
        PresetType.AGGRESSIVE: IndicatorPreset(
            name="Fast RSI",
            description="Quick 7-period RSI for rapid signals",
            parameters={"length": 7},
            recommended_timeframes=["1m", "5m", "15m"],
            use_case="Scalping and quick trades",
        ),
        PresetType.CONSERVATIVE: IndicatorPreset(
            name="Slow RSI",
            description="21-period RSI for reliable signals",
            parameters={"length": 21},
            recommended_timeframes=["4h", "1d", "1w"],
            use_case="Swing trading with fewer false signals",
        ),
    },
    "macd": {
        PresetType.DEFAULT: IndicatorPreset(
            name="Standard MACD",
            description="Classic 12/26/9 configuration",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            recommended_timeframes=["1h", "4h", "1d"],
            use_case="Trend following and momentum",
        ),
        PresetType.AGGRESSIVE: IndicatorPreset(
            name="Fast MACD",
            description="Faster 5/13/5 for quick signals",
            parameters={"fast": 5, "slow": 13, "signal": 5},
            recommended_timeframes=["5m", "15m", "1h"],
            use_case="Day trading and scalping",
        ),
        PresetType.CONSERVATIVE: IndicatorPreset(
            name="Slow MACD",
            description="Slower 19/39/9 for strong trends",
            parameters={"fast": 19, "slow": 39, "signal": 9},
            recommended_timeframes=["4h", "1d", "1w"],
            use_case="Position trading with confirmed trends",
        ),
    },
    "bollinger_bands": {
        PresetType.DEFAULT: IndicatorPreset(
            name="Standard Bollinger Bands",
            description="20-period with 2.0 standard deviations",
            parameters={"length": 20, "std": 2.0},
            recommended_timeframes=["15m", "1h", "4h", "1d"],
            use_case="Volatility-based trading",
        ),
        PresetType.AGGRESSIVE: IndicatorPreset(
            name="Tight Bollinger Bands",
            description="20-period with 1.5 standard deviations",
            parameters={"length": 20, "std": 1.5},
            recommended_timeframes=["5m", "15m", "1h"],
            use_case="Scalping with tighter bands",
        ),
        PresetType.CONSERVATIVE: IndicatorPreset(
            name="Wide Bollinger Bands",
            description="20-period with 2.5 standard deviations",
            parameters={"length": 20, "std": 2.5},
            recommended_timeframes=["4h", "1d"],
            use_case="Swing trading with wider tolerance",
        ),
    },
    "ema": {
        PresetType.SCALPING: IndicatorPreset(
            name="Scalping EMAs",
            description="Fast EMAs for scalping (8, 13, 21)",
            parameters={"periods": [8, 13, 21]},
            recommended_timeframes=["1m", "5m", "15m"],
            use_case="Quick trades with fast-moving averages",
        ),
        PresetType.DAY_TRADING: IndicatorPreset(
            name="Day Trading EMAs",
            description="Intraday EMAs (9, 21, 50)",
            parameters={"periods": [9, 21, 50]},
            recommended_timeframes=["15m", "1h", "4h"],
            use_case="Day trading with multiple timeframe confirmation",
        ),
        PresetType.SWING_TRADING: IndicatorPreset(
            name="Swing Trading EMAs",
            description="Swing EMAs (20, 50, 200)",
            parameters={"periods": [20, 50, 200]},
            recommended_timeframes=["4h", "1d"],
            use_case="Position trading with long-term trends",
        ),
    },
    "atr": {
        PresetType.DEFAULT: IndicatorPreset(
            name="Standard ATR",
            description="14-period ATR for volatility measurement",
            parameters={"length": 14},
            recommended_timeframes=["1h", "4h", "1d"],
            use_case="Stop loss and position sizing",
        ),
        PresetType.AGGRESSIVE: IndicatorPreset(
            name="Fast ATR",
            description="7-period ATR for rapid volatility changes",
            parameters={"length": 7},
            recommended_timeframes=["5m", "15m", "1h"],
            use_case="Scalping with tight stops",
        ),
    },
    "stochastic": {
        PresetType.DEFAULT: IndicatorPreset(
            name="Standard Stochastic",
            description="14/3/3 classic configuration",
            parameters={"k_period": 14, "d_period": 3, "smooth_k": 3},
            recommended_timeframes=["1h", "4h", "1d"],
            use_case="Overbought/oversold momentum trading",
        ),
        PresetType.AGGRESSIVE: IndicatorPreset(
            name="Fast Stochastic",
            description="5/3/3 for quick signals",
            parameters={"k_period": 5, "d_period": 3, "smooth_k": 3},
            recommended_timeframes=["5m", "15m", "1h"],
            use_case="Scalping with rapid reversals",
        ),
    },
}


# =====================================================================
# PARAMETER VALIDATION RULES
# =====================================================================

PARAMETER_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "length": {
        "type": "integer",
        "min": 1,
        "max": 500,
        "default": 20,
        "description": "Window length for calculation",
    },
    "periods": {
        "type": "array",
        "item_type": "integer",
        "min_items": 1,
        "max_items": 10,
        "item_min": 1,
        "item_max": 500,
        "default": [10, 20, 50],
        "description": "Multiple periods for calculation",
    },
    "fast": {
        "type": "integer",
        "min": 1,
        "max": 100,
        "default": 12,
        "description": "Fast period for MACD",
    },
    "slow": {
        "type": "integer",
        "min": 1,
        "max": 200,
        "default": 26,
        "description": "Slow period for MACD",
    },
    "signal": {
        "type": "integer",
        "min": 1,
        "max": 50,
        "default": 9,
        "description": "Signal line period",
    },
    "std": {
        "type": "float",
        "min": 0.1,
        "max": 5.0,
        "default": 2.0,
        "description": "Standard deviation multiplier",
    },
    "af": {
        "type": "float",
        "min": 0.001,
        "max": 1.0,
        "default": 0.02,
        "description": "Acceleration factor for Parabolic SAR",
    },
    "max_af": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "default": 0.2,
        "description": "Maximum acceleration factor",
    },
    "k_period": {
        "type": "integer",
        "min": 1,
        "max": 100,
        "default": 14,
        "description": "Stochastic %K period",
    },
    "d_period": {
        "type": "integer",
        "min": 1,
        "max": 50,
        "default": 3,
        "description": "Stochastic %D period",
    },
    "smooth_k": {
        "type": "integer",
        "min": 1,
        "max": 10,
        "default": 3,
        "description": "Smoothing for %K",
    },
    "levels": {
        "type": "integer",
        "min": 1,
        "max": 10,
        "default": 3,
        "description": "Number of support/resistance levels",
    },
    "timeperiod": {
        "type": "integer",
        "min": 1,
        "max": 100,
        "default": 1,
        "description": "Time period for pivot calculations",
    },
    "swing_length": {
        "type": "integer",
        "min": 1,
        "max": 50,
        "default": 5,
        "description": "Length for swing detection",
    },
}


# =====================================================================
# INDICATOR COMBINATIONS (STRATEGIES)
# =====================================================================


class IndicatorStrategy(BaseModel):
    """Predefined indicator combination strategy"""

    name: str
    description: str
    indicators: List[Dict[str, Any]]
    timeframes: List[str]
    risk_profile: str
    use_case: str


INDICATOR_STRATEGIES: Dict[str, IndicatorStrategy] = {
    "trend_following": IndicatorStrategy(
        name="Trend Following Strategy",
        description="Combine EMAs, MACD, and ADX for strong trend identification",
        indicators=[
            {
                "indicator_type": "ema",
                "parameters": {"periods": [20, 50, 200]},
                "overlay": True,
            },
            {
                "indicator_type": "macd",
                "parameters": {"fast": 12, "slow": 26, "signal": 9},
                "overlay": False,
            },
            {"indicator_type": "adx", "parameters": {"length": 14}, "overlay": False},
        ],
        timeframes=["1h", "4h", "1d"],
        risk_profile="Medium",
        use_case="Catching and riding strong trends",
    ),
    "mean_reversion": IndicatorStrategy(
        name="Mean Reversion Strategy",
        description="Use Bollinger Bands, RSI, and Stochastic for oversold/overbought conditions",
        indicators=[
            {
                "indicator_type": "bollinger_bands",
                "parameters": {"length": 20, "std": 2.0},
                "overlay": True,
            },
            {"indicator_type": "rsi", "parameters": {"length": 14}, "overlay": False},
            {
                "indicator_type": "stochastic",
                "parameters": {"k_period": 14, "d_period": 3, "smooth_k": 3},
                "overlay": False,
            },
        ],
        timeframes=["15m", "1h", "4h"],
        risk_profile="Medium-High",
        use_case="Trading reversals from extremes",
    ),
    "scalping": IndicatorStrategy(
        name="Scalping Strategy",
        description="Fast indicators for quick entries and exits",
        indicators=[
            {
                "indicator_type": "ema",
                "parameters": {"periods": [8, 13, 21]},
                "overlay": True,
            },
            {"indicator_type": "rsi", "parameters": {"length": 7}, "overlay": False},
            {"indicator_type": "atr", "parameters": {"length": 7}, "overlay": False},
            {"indicator_type": "vwap", "parameters": {}, "overlay": True},
        ],
        timeframes=["1m", "5m", "15m"],
        risk_profile="High",
        use_case="Quick trades with tight risk management",
    ),
    "smart_money": IndicatorStrategy(
        name="Smart Money Concepts Strategy",
        description="Institutional trading approach using market structure",
        indicators=[
            {"indicator_type": "fvg", "parameters": {}, "overlay": True},
            {"indicator_type": "order_blocks", "parameters": {}, "overlay": True},
            {"indicator_type": "liquidity_zones", "parameters": {}, "overlay": True},
            {"indicator_type": "break_of_structure", "parameters": {}, "overlay": True},
            {
                "indicator_type": "swing_highs_lows",
                "parameters": {"swing_length": 5},
                "overlay": True,
            },
        ],
        timeframes=["15m", "1h", "4h", "1d"],
        risk_profile="Medium",
        use_case="Trading like institutions with market structure",
    ),
    "volume_analysis": IndicatorStrategy(
        name="Volume Analysis Strategy",
        description="Volume-based confirmation and analysis",
        indicators=[
            {"indicator_type": "obv", "parameters": {}, "overlay": False},
            {"indicator_type": "vwap", "parameters": {}, "overlay": True},
            {
                "indicator_type": "volume_profile",
                "parameters": {"length": 20},
                "overlay": False,
            },
            {
                "indicator_type": "ema",
                "parameters": {"periods": [20, 50]},
                "overlay": True,
            },
        ],
        timeframes=["15m", "1h", "4h"],
        risk_profile="Medium",
        use_case="Volume confirmation for trend validation",
    ),
    "breakout": IndicatorStrategy(
        name="Breakout Strategy",
        description="Identify and trade breakouts with confirmation",
        indicators=[
            {
                "indicator_type": "bollinger_bands",
                "parameters": {"length": 20, "std": 2.0},
                "overlay": True,
            },
            {"indicator_type": "atr", "parameters": {"length": 14}, "overlay": False},
            {
                "indicator_type": "volume_profile",
                "parameters": {"length": 20},
                "overlay": False,
            },
            {
                "indicator_type": "support_resistance",
                "parameters": {"length": 20},
                "overlay": True,
            },
        ],
        timeframes=["1h", "4h", "1d"],
        risk_profile="Medium-High",
        use_case="Trading volatility breakouts",
    ),
}


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def get_preset(
    indicator_type: str, preset_type: PresetType
) -> Optional[IndicatorPreset]:
    """
    Get a preset configuration for an indicator

    Args:
        indicator_type: Type of indicator
        preset_type: Type of preset (default, aggressive, conservative, etc.)

    Returns:
        IndicatorPreset if found, None otherwise
    """
    if indicator_type in INDICATOR_PRESETS:
        return INDICATOR_PRESETS[indicator_type].get(preset_type)
    return None


def get_all_presets(indicator_type: str) -> Dict[PresetType, IndicatorPreset]:
    """Get all available presets for an indicator"""
    return INDICATOR_PRESETS.get(indicator_type, {})


def get_strategy(strategy_name: str) -> Optional[IndicatorStrategy]:
    """Get a predefined indicator strategy"""
    return INDICATOR_STRATEGIES.get(strategy_name)


def get_all_strategies() -> Dict[str, IndicatorStrategy]:
    """Get all available indicator strategies"""
    return INDICATOR_STRATEGIES


def validate_parameter(param_name: str, value: Any) -> tuple[bool, Optional[str]]:
    """
    Validate a parameter value against constraints

    Args:
        param_name: Name of the parameter
        value: Value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if param_name not in PARAMETER_CONSTRAINTS:
        return True, None

    constraint = PARAMETER_CONSTRAINTS[param_name]
    param_type = constraint["type"]

    # Type validation
    if param_type == "integer" and not isinstance(value, int):
        return False, f"Parameter '{param_name}' must be an integer"

    if param_type == "float" and not isinstance(value, (int, float)):
        return False, f"Parameter '{param_name}' must be a number"

    if param_type == "array" and not isinstance(value, list):
        return False, f"Parameter '{param_name}' must be an array"

    # Range validation
    if param_type in ["integer", "float"]:
        if "min" in constraint and value < constraint["min"]:
            return False, f"Parameter '{param_name}' must be >= {constraint['min']}"
        if "max" in constraint and value > constraint["max"]:
            return False, f"Parameter '{param_name}' must be <= {constraint['max']}"

    # Array validation
    if param_type == "array":
        if "min_items" in constraint and len(value) < constraint["min_items"]:
            return (
                False,
                f"Parameter '{param_name}' must have at least {constraint['min_items']} items",
            )
        if "max_items" in constraint and len(value) > constraint["max_items"]:
            return (
                False,
                f"Parameter '{param_name}' must have at most {constraint['max_items']} items",
            )

        # Validate array items
        if "item_min" in constraint or "item_max" in constraint:
            for item in value:
                if "item_min" in constraint and item < constraint["item_min"]:
                    return (
                        False,
                        f"Items in '{param_name}' must be >= {constraint['item_min']}",
                    )
                if "item_max" in constraint and item > constraint["item_max"]:
                    return (
                        False,
                        f"Items in '{param_name}' must be <= {constraint['item_max']}",
                    )

    return True, None


def get_default_parameters(indicator_type: str) -> Dict[str, Any]:
    """
    Get default parameters for an indicator type

    Args:
        indicator_type: Type of indicator

    Returns:
        Dictionary of default parameters
    """
    # Map indicator types to their typical parameters
    defaults_map = {
        "sma": {"length": 20},
        "ema": {"length": 20},
        "wma": {"length": 20},
        "moving_average": {"periods": [10, 20, 50, 100, 200]},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "rsi": {"length": 14},
        "bollinger_bands": {"length": 20, "std": 2.0},
        "atr": {"length": 14},
        "stochastic": {"k_period": 14, "d_period": 3, "smooth_k": 3},
        "adx": {"length": 14},
        "cci": {"length": 20},
        "williams_r": {"length": 14},
        "parabolic_sar": {"af": 0.02, "max_af": 0.2},
        "pivot_points": {"timeperiod": 1, "levels": 3},
        "fibonacci": {"length": 50},
        "support_resistance": {"length": 20},
        "swing_highs_lows": {"swing_length": 5},
        "volume_profile": {"length": 20},
        "keltner_channels": {"length": 20, "std": 2.0},
    }

    return defaults_map.get(indicator_type, {})


def get_required_data_points(indicator_type: str, parameters: Dict[str, Any]) -> int:
    """
    Calculate minimum required data points for an indicator

    Args:
        indicator_type: Type of indicator
        parameters: Indicator parameters

    Returns:
        Minimum number of data points required
    """
    # Base requirements
    if indicator_type in ["sma", "ema", "wma"]:
        return parameters.get("length", 20)

    elif indicator_type == "macd":
        return max(
            parameters.get("fast", 12), parameters.get("slow", 26)
        ) + parameters.get("signal", 9)

    elif indicator_type == "bollinger_bands":
        return parameters.get("length", 20) + 10  # Extra for std calculation

    elif indicator_type == "rsi":
        length = parameters.get("length", 14)
        if isinstance(length, list):
            return max(length) + 10
        return length + 10

    elif indicator_type == "stochastic":
        return parameters.get("k_period", 14) + parameters.get("smooth_k", 3)

    elif indicator_type == "atr":
        return parameters.get("length", 14) + 5

    elif indicator_type in [
        "fvg",
        "order_blocks",
        "liquidity_zones",
        "break_of_structure",
    ]:
        return 50  # SMC indicators need more data

    # Default minimum
    return 30


# =====================================================================
# INDICATOR METADATA
# =====================================================================

INDICATOR_METADATA = {
    "categories": {
        "trend": ["sma", "ema", "wma", "macd", "parabolic_sar", "adx", "ichimoku"],
        "momentum": ["rsi", "stochastic", "cci", "williams_r"],
        "volatility": ["bollinger_bands", "atr", "keltner_channels"],
        "volume": ["obv", "vwap", "volume_profile"],
        "support_resistance": [
            "pivot_points",
            "fibonacci",
            "trendlines",
            "support_resistance",
        ],
        "smc": [
            "fvg",
            "order_blocks",
            "liquidity_zones",
            "break_of_structure",
            "swing_highs_lows",
        ],
        "patterns": ["heikin_ashi", "doji", "engulfing", "hammer", "shooting_star"],
    },
    "overlay_indicators": [
        "sma",
        "ema",
        "wma",
        "bollinger_bands",
        "parabolic_sar",
        "pivot_points",
        "fibonacci",
        "trendlines",
        "support_resistance",
        "vwap",
        "fvg",
        "order_blocks",
        "liquidity_zones",
        "break_of_structure",
        "swing_highs_lows",
        "ichimoku",
        "keltner_channels",
    ],
    "oscillators": ["rsi", "stochastic", "macd", "cci", "williams_r", "obv"],
}


def is_overlay_indicator(indicator_type: str) -> bool:
    """Check if indicator should overlay on price chart"""
    return indicator_type in INDICATOR_METADATA["overlay_indicators"]


def is_oscillator(indicator_type: str) -> bool:
    """Check if indicator is an oscillator"""
    return indicator_type in INDICATOR_METADATA["oscillators"]


def get_indicator_category(indicator_type: str) -> Optional[str]:
    """Get the category of an indicator"""
    for category, indicators in INDICATOR_METADATA["categories"].items():
        if indicator_type in indicators:
            return category
    return None
