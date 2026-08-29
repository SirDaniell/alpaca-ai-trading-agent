"""
Smart Money Concepts (SMC) - Market Structure Analysis Engine

This module provides algorithmic detection of institutional trading patterns and market structure
using Smart Money Concepts (SMC) methodology based on Inner Circle Trader (ICT) principles.

ATTRIBUTION:
    This module is derived from the MIT-licensed smart-money-concepts project:
    - Source: https://github.com/joshyattridge/smart-money-concepts
    - Author: Joshua Attridge (@joshyattridge)
    - License: MIT
    - Original PyPI: https://pypi.org/project/smartmoneyconcepts/

FEATURES:
    - Fair Value Gap (FVG) detection
    - Swing highs and lows identification
    - Break of Structure (BOS) and Change of Character (CHoCH)
    - Order Block (OB) detection
    - Liquidity zone mapping
    - Session-based analysis (Sydney, Tokyo, London, New York, custom)
    - Retracement level calculation
    - Previous high/low tracking

INTEGRATION:
    All methods are fully vectorized and return Pandas DataFrames for seamless integration
    into the fin-dash-buddy analysis pipeline.

DISCLAIMER:
    This module is for educational purposes only. Do not use as a sole decision maker for trades.
    Always use proper risk management and do your own research before trading.
"""

from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime

# --- Decorator Definitions ---


def inputvalidator(input_="ohlc"):
    """
    Decorator factory to create an input validation decorator for DataFrame methods.
    It ensures that the input DataFrame contains specified OHLCV columns (open, high, low, close, volume).
    It also standardizes column names to lowercase.

    Args:
        input_ (str): A string representing the required columns.
                      'o' for open, 'h' for high, 'l' for low, 'c' for close, 'v' for volume.
                      Defaults to "ohlc".

    Returns:
        function: A decorator function that can be applied to class methods.
    """

    def dfcheck(func):
        """
        The actual decorator that wraps the target function.
        It performs input validation and column renaming before calling the original function.
        """

        @wraps(func)
        def wrap(*args, **kwargs):
            """
            The wrapper function that executes the validation logic.
            """
            # Convert args to a mutable list to allow modification (e.g., renaming columns)
            args_list = list(args)

            # Determine the index of the DataFrame argument.
            # Assumes the DataFrame is either the first argument (self/cls) or the second.
            df_arg_index = 0 if isinstance(args_list[0], pd.DataFrame) else 1

            # Rename DataFrame columns to lowercase for consistent access
            args_list[df_arg_index] = args_list[df_arg_index].rename(
                columns={c: c.lower() for c in args_list[df_arg_index].columns}
            )

            # Define a mapping from shorthand input codes to full column names
            inputs_map = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get(
                    "column", "close"
                ).lower(),  # 'c' can be customized via 'column' kwarg
                "v": "volume",
            }

            # If a custom 'close' column name was provided, update kwargs
            if inputs_map["c"] != "close":
                kwargs["column"] = inputs_map["c"]

            # Check if all required columns are present in the DataFrame
            for required_char in input_:
                required_col_name = inputs_map[required_char]
                if required_col_name not in args_list[df_arg_index].columns:
                    raise LookupError(
                        f'Must have a DataFrame column named "{required_col_name}"'
                    )

            # Call the original function with the modified arguments
            return func(*args_list, **kwargs)

        return wrap

    return dfcheck


# The 'apply' decorator is removed to improve picklability.
# Instead, @inputvalidator will be applied directly to each class method.
# def apply(decorator):
#     def decorate(cls):
#         for attr in cls.__dict__:
#             if callable(getattr(cls, attr)):
#                 setattr(cls, attr, decorator(getattr(cls, attr)))
#         return cls
#     return decorate


# --- SMC Class Definition ---


# The @apply decorator is removed here.
# @apply(inputvalidator(input_="ohlc"))
class smc:
    """
    A class for calculating Smart Money Concepts (SMC) and other market structure indicators.
    All methods are class methods and operate on pandas DataFrames.
    """

    __version__ = "0.0.26"

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def fvg(
        cls,
        ohlc: DataFrame,
        join_consecutive=False,
        min_candle_size_ratio: float = 0.75,
        candle_lookback: int = 3,
    ) -> DataFrame:
        """
        Calculates Fair Value Gaps (FVG).
        A Fair Value Gap is a price inefficiency where there's a gap between the previous high/low
        and the next low/high, indicating an imbalance in supply/demand.

        Args:
            ohlc (DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
            join_consecutive (bool): If True, consecutive FVGs of the same type are merged
                                     into a single, larger FVG.
            min_candle_size_ratio (float): Minimum middle-candle high-low range relative
                                           to the average range of the previous candles.
                                           Smaller candles are ignored as noise.
            candle_lookback (int): Number of preceding candle ranges used for the size
                                   comparison.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - FVG (int): 1 for bullish FVG, -1 for bearish FVG, NaN otherwise.
                - Top (float): The upper boundary of the FVG.
                - Bottom (float): The lower boundary of the FVG.
                - MitigatedIndex (int): The index of the candle that mitigated (filled) the FVG.
                                        NaN if not mitigated.
        """
        if min_candle_size_ratio < 0:
            raise ValueError("min_candle_size_ratio must be non-negative")
        if candle_lookback < 1:
            raise ValueError("candle_lookback must be at least 1")

        # Reject small middle candles before detecting gaps. The FVG pattern can
        # still occur on a tiny candle, but those records are usually noise.
        candle_ranges = (ohlc["high"] - ohlc["low"]).to_numpy(dtype=float)
        previous_range_average = (
            pd.Series(candle_ranges)
            .rolling(window=candle_lookback, min_periods=candle_lookback)
            .mean()
            .shift(1)
            .to_numpy()
        )
        candle_size_ok = (
            np.isfinite(previous_range_average)
            & (candle_ranges >= previous_range_average * min_candle_size_ratio)
        )

        # Determine FVG presence and type using 3-candle structural gap condition:
        # Bullish FVG: high(Candle 1) < low(Candle 3)
        # Bearish FVG: low(Candle 1) > high(Candle 3)
        is_bullish_gap = (
            (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
            & (ohlc["close"] > ohlc["open"])
            & candle_size_ok
        )
        is_bearish_gap = (
            (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
            & (ohlc["close"] < ohlc["open"])
            & candle_size_ok
        )

        fvg = np.where(
            is_bullish_gap,
            1,
            np.where(is_bearish_gap, -1, np.nan),
        )

        # Determine the 'Top' level of the FVG:
        # For bullish FVG: low of candle 3 (low.shift(-1))
        # For bearish FVG: low of candle 1 (low.shift(1))
        top = np.where(
            ~np.isnan(fvg),
            np.where(
                fvg == 1,
                ohlc["low"].shift(-1),
                ohlc["low"].shift(1),
            ),
            np.nan,
        )

        # Determine the 'Bottom' level of the FVG:
        # For bullish FVG: high of candle 1 (high.shift(1))
        # For bearish FVG: high of candle 3 (high.shift(-1))
        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                fvg == 1,
                ohlc["high"].shift(1),
                ohlc["high"].shift(-1),
            ),
            np.nan,
        )

        # Optional: Join consecutive FVGs
        if join_consecutive:
            # Iterate through the FVG array to find and merge consecutive gaps
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:  # If current and next FVG are of the same type
                    # Merge by taking the highest top and lowest bottom
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    # Mark the current FVG as NaN since it's merged into the next
                    fvg[i] = top[i] = bottom[i] = np.nan

        # Calculate mitigation index for each FVG
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        # Iterate only over indices where an FVG was identified
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:  # Bullish FVG: require a close through the lower boundary
                mask = ohlc["close"].to_numpy()[i + 2 :] <= bottom[i]
            elif fvg[i] == -1:  # Bearish FVG: require a close through the upper boundary
                mask = ohlc["close"].to_numpy()[i + 2 :] >= top[i]

            if np.any(mask):
                # Find the first index where mitigation occurs
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        # Set mitigated_index to NaN for non-FVG candles
        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        # Concatenate all calculated Series into a single DataFrame and return
        return pd.concat(
            [
                pd.Series(
                    fvg, name="FVG", index=ohlc.index
                ),  # Ensure index is preserved
                pd.Series(top, name="Top", index=ohlc.index),
                pd.Series(bottom, name="Bottom", index=ohlc.index),
                pd.Series(mitigated_index, name="MitigatedIndex", index=ohlc.index),
            ],
            axis=1,
        )

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
        """
        Identifies Swing highs and Swing lows based on a lookback and lookforward window.
        A swing high is the highest high within a specified window before and after the current candle.
        A swing low is the lowest low within a specified window before and after the current candle.

        Args:
            ohlc (DataFrame): DataFrame with 'high' and 'low' columns.
            swing_length (int): The number of candles to look back and forward.
                                The total window size will be `2 * swing_length + 1`.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - highlow (int): 1 if swing high, -1 if swing low, NaN otherwise.
                - Level (float): The price level of the swing high or low.
        """
        # Match the reference implementation's lookaround window exactly.
        # After doubling the requested length, shifting by half the window
        # makes the rolling result cover the same bars before and after the
        # candidate swing.
        total_swing_window = swing_length * 2
        half_window = total_swing_window // 2

        is_swing_high = (
            ohlc["high"]
            == ohlc["high"].shift(-half_window).rolling(total_swing_window).max()
        )
        is_swing_low = (
            ohlc["low"]
            == ohlc["low"].shift(-half_window).rolling(total_swing_window).min()
        )

        # Combine into a single array: 1 for swing high, -1 for swing low, NaN otherwise
        swing_highs_lows = np.where(
            is_swing_high,
            1,
            np.where(is_swing_low, -1, np.nan),
        )

        # --- Refinement Loop to remove consecutive swings of the same type ---
        # This loop iteratively removes redundant swing points.
        # For consecutive highs, keep the higher one. For consecutive lows, keep the lower one.
        while True:
            # Get indices of all identified swing points
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                # No more pairs to compare, or only one swing point left
                break

            # Get the swing types and levels for current and next swing points
            current_swings = swing_highs_lows[positions[:-1]]
            next_swings = swing_highs_lows[positions[1:]]

            current_highs = ohlc["high"].iloc[positions[:-1]].values
            current_lows = ohlc["low"].iloc[positions[:-1]].values

            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            # Boolean array to mark indices for removal
            index_to_remove = np.zeros(len(positions), dtype=bool)

            # Identify consecutive highs (1, 1)
            consecutive_highs = (current_swings == 1) & (next_swings == 1)
            # If current high is lower than next high, mark current for removal
            index_to_remove[:-1] |= consecutive_highs & (current_highs < next_highs)
            # If current high is greater than or equal to next high, mark next for removal
            index_to_remove[1:] |= consecutive_highs & (current_highs >= next_highs)

            # Identify consecutive lows (-1, -1)
            consecutive_lows = (current_swings == -1) & (next_swings == -1)
            # If current low is higher than next low, mark current for removal
            index_to_remove[:-1] |= consecutive_lows & (current_lows > next_lows)
            # If current low is less than or equal to next low, mark next for removal
            index_to_remove[1:] |= consecutive_lows & (current_lows <= next_lows)

            # If no indices were marked for removal in this iteration, stop
            if not index_to_remove.any():
                break

            # Set the marked swing points to NaN
            swing_highs_lows[positions[index_to_remove]] = np.nan

        # --- Handle edge cases: First and Last Swing Points ---
        # This logic ensures that the first and last identified swing points
        # alternate in type (e.g., if the first is a high, the very first bar is marked as a low).
        # This is a common convention in some SMC methodologies.
        positions = np.where(~np.isnan(swing_highs_lows))[0]
        if len(positions) > 0:
            # If the first identified swing is a high, force the very first bar to be a low
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            # If the first identified swing is a low, force the very first bar to be a high
            elif swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1

            # Similar logic for the last identified swing point
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            elif swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        # Determine the price level for each identified swing point
        level = np.where(
            ~np.isnan(swing_highs_lows),  # Only for identified swing points
            np.where(
                swing_highs_lows == 1, ohlc["high"], ohlc["low"]
            ),  # Use high for swing high, low for swing low
            np.nan,
        )

        # Concatenate results into a DataFrame and return
        return pd.concat(
            [
                pd.Series(
                    swing_highs_lows, name="highlow", index=ohlc.index
                ),  # Preserve original index
                pd.Series(level, name="Level", index=ohlc.index),
            ],
            axis=1,
        )

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
    ) -> DataFrame:
        """
        Identifies Break of Structure (BOS) and Change of Character (CHoCH) events.
        These are key concepts in Smart Money Concepts, indicating shifts in market trend.

        - BOS: Continuation of the current trend (e.g., higher high in an uptrend).
        - CHoCH: Reversal of the current trend (e.g., lower low after an uptrend).

        Args:
            ohlc (DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
            swing_highs_lows (DataFrame): DataFrame containing 'highlow' and 'Level'
                                          from `smc.swing_highs_lows` function.
            close_break (bool): If True, a break is confirmed by the closing price of a candle.
                                If False, the high/low of the candle is used.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - BOS (int): 1 for bullish BOS, -1 for bearish BOS, NaN otherwise.
                - CHOCH (int): 1 for bullish CHoCH, -1 for bearish CHoCH, NaN otherwise.
                - Level (float): The price level that was broken.
                - BrokenIndex (int): The index of the candle that confirmed the break.
                                     NaN if not broken.
        """
        # Ensure working on copies to avoid modifying original DataFrames
        swing_highs_lows = swing_highs_lows.copy()

        # Initialize arrays for storing results
        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level_broken = np.zeros(
            len(ohlc), dtype=np.float32
        )  # Renamed to avoid conflict with 'Level' column
        broken_index = np.zeros(len(ohlc), dtype=np.int32)

        # Lists to keep track of swing levels and types in order of occurrence
        level_order = []
        highs_lows_order = []
        last_positions = []  # Stores the original index of the swing points

        # Iterate through the swing_highs_lows DataFrame to build the swing sequence
        for i in range(len(swing_highs_lows["highlow"])):
            if not np.isnan(
                swing_highs_lows["highlow"].iloc[i]
            ):  # Check if it's a valid swing point
                level_order.append(swing_highs_lows["Level"].iloc[i])
                highs_lows_order.append(swing_highs_lows["highlow"].iloc[i])
                last_positions.append(i)  # Store the original index

                # Check for BOS/CHoCH patterns once at least 4 swing points are recorded
                if len(level_order) >= 4:
                    # Extract the last 4 swing levels and types
                    # These represent the sequence needed to identify BOS/CHoCH
                    # Example: [-1, 1, -1, 1] means low, high, low, high (potential bullish structure)

                    # --- Bullish BOS (Break of Structure) ---
                    # Pattern: low, high, lower low, higher high (bullish continuation)
                    # Levels: L1 < L3 < L2 < L4 (where L1 is level_order[-4], L2 is level_order[-3], etc.)
                    is_bullish_bos = (
                        np.all(
                            highs_lows_order[-4:] == [-1, 1, -1, 1]
                        )  # Swing types sequence
                        and (
                            level_order[-4] < level_order[-2]
                        )  # Previous low is lower than current low
                        and (
                            level_order[-2] < level_order[-3]
                        )  # Current low is lower than previous high
                        and (
                            level_order[-3] < level_order[-1]
                        )  # Previous high is lower than current high
                    )
                    if is_bullish_bos:
                        bos[last_positions[-2]] = (
                            1  # Mark BOS at the second to last swing point's original index
                        )
                        level_broken[last_positions[-2]] = level_order[
                            -3
                        ]  # The level broken is the previous high

                    # --- Bearish BOS (Break of Structure) ---
                    # Pattern: high, low, higher high, lower low (bearish continuation)
                    # Levels: H1 > H3 > H2 > H4
                    is_bearish_bos = (
                        np.all(
                            highs_lows_order[-4:] == [1, -1, 1, -1]
                        )  # Swing types sequence
                        and (
                            level_order[-4] > level_order[-2]
                        )  # Previous high is higher than current high
                        and (
                            level_order[-2] > level_order[-3]
                        )  # Current high is higher than previous low
                        and (
                            level_order[-3] > level_order[-1]
                        )  # Previous low is higher than current low
                    )
                    if is_bearish_bos:
                        bos[last_positions[-2]] = -1  # Mark BOS
                        level_broken[last_positions[-2]] = level_order[
                            -3
                        ]  # The level broken is the previous low

                    # --- Bullish CHoCH (Change of Character) ---
                    # Pattern: low, high, lower low, higher high (bearish to bullish reversal)
                    # Levels: L4 > L2 > L1 > L3 (where L1 is level_order[-4], L2 is level_order[-3], etc.)
                    is_bullish_choch = (
                        np.all(
                            highs_lows_order[-4:] == [-1, 1, -1, 1]
                        )  # Swing types sequence
                        and (
                            level_order[-1] > level_order[-3]
                        )  # Current high is higher than previous high
                        and (
                            level_order[-3] > level_order[-4]
                        )  # Previous high is higher than previous low
                        and (
                            level_order[-4] > level_order[-2]
                        )  # Previous low is higher than current low
                    )
                    if is_bullish_choch:
                        choch[last_positions[-2]] = 1  # Mark CHoCH
                        # If a BOS was already marked at this position, keep its level. Otherwise, use the CHoCH level.
                        level_broken[last_positions[-2]] = (
                            level_order[-3]
                            if bos[last_positions[-2]] == 0
                            else level_broken[last_positions[-2]]
                        )

                    # --- Bearish CHoCH (Change of Character) ---
                    # Pattern: high, low, higher high, lower low (bullish to bearish reversal)
                    # Levels: H4 < H2 < H1 < H3
                    is_bearish_choch = (
                        np.all(
                            highs_lows_order[-4:] == [1, -1, 1, -1]
                        )  # Swing types sequence
                        and (
                            level_order[-1] < level_order[-3]
                        )  # Current low is lower than previous low
                        and (
                            level_order[-3] < level_order[-4]
                        )  # Previous low is lower than previous high
                        and (
                            level_order[-4] < level_order[-2]
                        )  # Previous high is lower than current high
                    )
                    if is_bearish_choch:
                        choch[last_positions[-2]] = -1  # Mark CHoCH
                        # If a BOS was already marked, keep its level. Otherwise, use the CHoCH level.
                        level_broken[last_positions[-2]] = (
                            level_order[-3]
                            if bos[last_positions[-2]] == 0
                            else level_broken[last_positions[-2]]
                        )

        # --- Determine the candle that broke the level (BrokenIndex) ---
        # Iterate over all identified BOS/CHoCH events
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)

            # Check for bullish break (BOS=1 or CHoCH=1)
            if bos[i] == 1 or choch[i] == 1:
                # Look for a candle (starting from 2 candles after the event)
                # whose close (or high) is greater than the broken level
                check_col = "close" if close_break else "high"
                mask = ohlc[check_col][i + 2 :] > level_broken[i]
            # Check for bearish break (BOS=-1 or CHoCH=-1)
            elif bos[i] == -1 or choch[i] == -1:
                # Look for a candle (starting from 2 candles after the event)
                # whose close (or low) is less than the broken level
                check_col = "close" if close_break else "low"
                mask = ohlc[check_col][i + 2 :] < level_broken[i]

            if np.any(mask):
                # Find the index of the first candle that breaks the level
                j = np.argmax(mask) + i + 2
                broken_index[i] = j

                # --- Invalidation of previous unbroken BOS/CHoCH ---
                # If a new break occurs, any *earlier* BOS/CHoCH events that were
                # expected to break *after* this new break are now considered invalid.
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    # Check if 'k' is an earlier event AND its 'broken_index' was after or at 'j'
                    if k < i and broken_index[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level_broken[k] = 0
                        broken_index[k] = 0  # Also reset its broken index

        # --- Remove unconfirmed BOS/CHoCH events ---
        # Any BOS/CHoCH events that were identified but never had a confirming 'broken_index'
        # are set to NaN (or 0) to indicate they were not valid breaks.
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken_index == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level_broken[i] = 0
            broken_index[i] = 0

        # Replace all 0s (which mean 'no event' or 'invalidated event') with NaN for cleaner output
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level_broken = np.where(level_broken != 0, level_broken, np.nan)
        broken_index = np.where(broken_index != 0, broken_index, np.nan)

        # Concatenate results into a DataFrame and return
        return pd.concat(
            [
                pd.Series(bos, name="BOS", index=ohlc.index),  # Preserve original index
                pd.Series(choch, name="CHOCH", index=ohlc.index),
                pd.Series(level_broken, name="Level", index=ohlc.index),
                pd.Series(broken_index, name="BrokenIndex", index=ohlc.index),
            ],
            axis=1,
        )

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def ob(
        cls,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_mitigation: bool = False,
    ) -> DataFrame:
        """
        Detects Order Blocks (OBs), which are price ranges where a significant amount of
        institutional orders are believed to be placed, often leading to price reactions.

        Args:
            ohlc (DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
            swing_highs_lows (DataFrame): DataFrame containing 'highlow' and 'Level'
                                          from `smc.swing_highs_lows` function.
            close_mitigation (bool): If True, an order block is considered mitigated if
                                     the candle's close price crosses the OB boundary.
                                     If False, the high/low of the candle is used.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - OB (int): 1 if bullish OB, -1 if bearish OB, NaN otherwise.
                - Top (float): The upper boundary of the order block.
                - Bottom (float): The lower boundary of the order block.
                - OBvolume (float): Sum of volume of the OB candle and the two preceding candles.
                - Percentage (float): Strength of the order block (min(highvolume, lowvolume)/max(highvolume, lowvolume) * 100).
                - MitigatedIndex (int): The index of the candle that mitigated the order block.
                                        NaN if not mitigated.
        """
        ohlc_len = len(ohlc)
        # Extract numpy arrays for faster access
        _open = ohlc["open"].values
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values
        _volume = ohlc["volume"].values
        swing_hl = swing_highs_lows["highlow"].values

        # Pre-allocate arrays for results, initialized with zeros or False
        ob = np.zeros(ohlc_len, dtype=np.int32)
        top_arr = np.zeros(ohlc_len, dtype=np.float32)
        bottom_arr = np.zeros(ohlc_len, dtype=np.float32)
        obvolume = np.zeros(ohlc_len, dtype=np.float32)
        lowvolume = np.zeros(ohlc_len, dtype=np.float32)
        highvolume = np.zeros(ohlc_len, dtype=np.float32)
        percentage = np.zeros(ohlc_len, dtype=np.float32)
        mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
        breaker = np.full(
            ohlc_len, False, dtype=bool
        )  # Tracks if an OB has been 'broken' (crossed for mitigation check)
        crossed = np.full(
            ohlc_len, False, dtype=bool
        )  # Tracks if a swing high/low has been 'crossed' to trigger OB detection

        # Precompute swing high/low indices for efficient lookup
        swing_high_indices = np.flatnonzero(swing_hl == 1)
        swing_low_indices = np.flatnonzero(swing_hl == -1)

        # --- Bullish Order Block Detection and Mitigation ---
        active_bullish = []  # List to store indices of active (unmitigated) bullish OBs
        for i in range(ohlc_len):
            current_index = i

            # 1. Check for mitigation of active bullish OBs
            # An active bullish OB is mitigated if price drops below its bottom boundary.
            for (
                idx
            ) in (
                active_bullish.copy()
            ):  # Iterate over a copy to allow modification during loop
                if breaker[
                    idx
                ]:  # If the OB has already been 'broken' (crossed for the first time)
                    # If current high goes above the OB's top, it's fully invalidated
                    if _high[current_index] > top_arr[idx]:
                        # Reset all values for this invalidated OB
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obvolume[idx] = 0.0
                        lowvolume[idx] = 0.0
                        highvolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bullish.remove(idx)  # Remove from active list
                else:  # If the OB has not yet been 'broken'
                    # Check if current low (or close) crosses the OB's bottom
                    if (
                        not close_mitigation and _low[current_index] < bottom_arr[idx]
                    ) or (  # low crosses bottom
                        close_mitigation
                        and min(_open[current_index], _close[current_index])
                        < bottom_arr[idx]
                    ):  # close crosses bottom
                        breaker[idx] = True  # Mark as broken
                        mitigated_index[idx] = current_index  # Record mitigation index

            # 2. Detect new bullish Order Blocks
            # A bullish OB forms after a swing high is broken to the upside.
            # Find the last swing high that occurred before the current candle
            pos = np.searchsorted(swing_high_indices, current_index)
            last_top_index = swing_high_indices[pos - 1] if pos > 0 else None

            if last_top_index is not None:
                # If current close breaks above the last swing high AND that swing high hasn't been crossed yet
                if (
                    _close[current_index] > _high[last_top_index]
                    and not crossed[last_top_index]
                ):
                    crossed[last_top_index] = True  # Mark this swing high as crossed

                    # Default OB candle is the candle before the current one
                    default_ob_index = current_index - 1
                    # Ensure default_ob_index is valid
                    if default_ob_index < 0:
                        continue  # Skip if not enough history

                    # Initial assumption for OB boundaries (candle before the break)
                    ob_bottom = _low[default_ob_index]
                    ob_top = _high[default_ob_index]
                    ob_candle_index = default_ob_index

                    # Look for the lowest low between the last swing high and the current candle
                    # This lowest low candle forms the potential bullish order block.
                    if (
                        current_index - last_top_index > 1
                    ):  # Ensure there's a segment to check
                        start_segment = last_top_index + 1
                        end_segment = current_index
                        if end_segment > start_segment:
                            segment_lows = _low[start_segment:end_segment]
                            min_val_in_segment = segment_lows.min()
                            # Find the last occurrence of this minimum value (important for multiple lows)
                            candidates_indices = np.nonzero(
                                segment_lows == min_val_in_segment
                            )[0]
                            if candidates_indices.size:
                                candidate_index_in_ohlc = (
                                    start_segment + candidates_indices[-1]
                                )
                                ob_bottom = _low[candidate_index_in_ohlc]
                                ob_top = _high[candidate_index_in_ohlc]
                                ob_candle_index = candidate_index_in_ohlc

                    # Set bullish OB values
                    ob[ob_candle_index] = 1
                    top_arr[ob_candle_index] = ob_top
                    bottom_arr[ob_candle_index] = ob_bottom

                    # Calculate OB volume (current + 2 preceding volumes)
                    # Ensure indices are valid
                    vol1 = _volume[current_index] if current_index >= 0 else 0
                    vol2 = _volume[current_index - 1] if current_index - 1 >= 0 else 0
                    vol3 = _volume[current_index - 2] if current_index - 2 >= 0 else 0
                    obvolume[ob_candle_index] = vol1 + vol2 + vol3

                    # Calculate high and low volumes for percentage calculation
                    highvolume[ob_candle_index] = vol1 + vol2
                    lowvolume[ob_candle_index] = vol3

                    max_vol = max(
                        highvolume[ob_candle_index], lowvolume[ob_candle_index]
                    )
                    percentage[ob_candle_index] = (
                        (
                            (
                                min(
                                    highvolume[ob_candle_index],
                                    lowvolume[ob_candle_index],
                                )
                                / max_vol
                            )
                            * 100.0
                        )
                        if max_vol != 0
                        else 100.0
                    )  # Handle division by zero

                    active_bullish.append(ob_candle_index)  # Add to active list

        # --- Bearish Order Block Detection and Mitigation ---
        active_bearish = []  # List to store indices of active (unmitigated) bearish OBs
        for i in range(ohlc_len):
            current_index = i

            # 1. Check for mitigation of active bearish OBs
            # An active bearish OB is mitigated if price rises above its top boundary.
            for idx in active_bearish.copy():
                if breaker[idx]:  # If the OB has already been 'broken'
                    # If current low goes below the OB's bottom, it's fully invalidated
                    if _low[current_index] < bottom_arr[idx]:
                        # Reset all values for this invalidated OB
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obvolume[idx] = 0.0
                        lowvolume[idx] = 0.0
                        highvolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bearish.remove(idx)
                else:  # If the OB has not yet been 'broken'
                    # Check if current high (or close) crosses the OB's top
                    if (
                        not close_mitigation and _high[current_index] > top_arr[idx]
                    ) or (  # high crosses top
                        close_mitigation
                        and max(_open[current_index], _close[current_index])
                        > top_arr[idx]
                    ):  # close crosses top
                        breaker[idx] = True
                        mitigated_index[idx] = current_index  # Record mitigation index

            # 2. Detect new bearish Order Blocks
            # A bearish OB forms after a swing low is broken to the downside.
            # Find the last swing low that occurred before the current candle
            pos = np.searchsorted(swing_low_indices, current_index)
            last_btm_index = swing_low_indices[pos - 1] if pos > 0 else None

            if last_btm_index is not None:
                # If current close breaks below the last swing low AND that swing low hasn't been crossed yet
                if (
                    _close[current_index] < _low[last_btm_index]
                    and not crossed[last_btm_index]
                ):
                    crossed[last_btm_index] = True  # Mark this swing low as crossed

                    # Default OB candle is the candle before the current one
                    default_ob_index = current_index - 1
                    if default_ob_index < 0:
                        continue  # Skip if not enough history

                    # Initial assumption for OB boundaries
                    ob_top = _high[default_ob_index]
                    ob_bottom = _low[default_ob_index]
                    ob_candle_index = default_ob_index

                    # Look for the highest high between the last swing low and the current candle
                    # This highest high candle forms the potential bearish order block.
                    if current_index - last_btm_index > 1:
                        start_segment = last_btm_index + 1
                        end_segment = current_index
                        if end_segment > start_segment:
                            segment_highs = _high[start_segment:end_segment]
                            max_val_in_segment = segment_highs.max()
                            # Find the last occurrence of this maximum value
                            candidates_indices = np.nonzero(
                                segment_highs == max_val_in_segment
                            )[0]
                            if candidates_indices.size:
                                candidate_index_in_ohlc = (
                                    start_segment + candidates_indices[-1]
                                )
                                ob_top = _high[candidate_index_in_ohlc]
                                ob_bottom = _low[candidate_index_in_ohlc]
                                ob_candle_index = candidate_index_in_ohlc

                    # Set bearish OB values
                    ob[ob_candle_index] = -1
                    top_arr[ob_candle_index] = ob_top
                    bottom_arr[ob_candle_index] = ob_bottom

                    # Calculate OB volume
                    vol1 = _volume[current_index] if current_index >= 0 else 0
                    vol2 = _volume[current_index - 1] if current_index - 1 >= 0 else 0
                    vol3 = _volume[current_index - 2] if current_index - 2 >= 0 else 0
                    obvolume[ob_candle_index] = vol1 + vol2 + vol3

                    # Calculate high and low volumes for percentage
                    lowvolume[ob_candle_index] = vol1 + vol2
                    highvolume[ob_candle_index] = vol3

                    max_vol = max(
                        highvolume[ob_candle_index], lowvolume[ob_candle_index]
                    )
                    percentage[ob_candle_index] = (
                        (
                            (
                                min(
                                    highvolume[ob_candle_index],
                                    lowvolume[ob_candle_index],
                                )
                                / max_vol
                            )
                            * 100.0
                        )
                        if max_vol != 0
                        else 100.0
                    )

                    active_bearish.append(ob_candle_index)

        # Convert 0s (no OB or invalidated OB) to NaN for final output
        ob = np.where(ob != 0, ob, np.nan)
        top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
        bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
        obvolume = np.where(~np.isnan(ob), obvolume, np.nan)
        mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
        percentage = np.where(~np.isnan(ob), percentage, np.nan)

        # Concatenate all result arrays into a DataFrame and return
        return pd.concat(
            [
                pd.Series(ob, name="OB", index=ohlc.index),
                pd.Series(top_arr, name="Top", index=ohlc.index),
                pd.Series(bottom_arr, name="Bottom", index=ohlc.index),
                pd.Series(obvolume, name="OBvolume", index=ohlc.index),
                pd.Series(mitigated_index, name="MitigatedIndex", index=ohlc.index),
                pd.Series(percentage, name="Percentage", index=ohlc.index),
            ],
            axis=1,
        )

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def liquidity(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
    ) -> DataFrame:
        """
        Detects Liquidity zones, which are areas where multiple swing highs or lows
        are clustered within a small price range, indicating potential areas for
        stop-loss hunting or order accumulation.

        Args:
            ohlc (DataFrame): DataFrame with 'high' and 'low' columns.
            swing_highs_lows (DataFrame): DataFrame containing 'highlow' and 'Level'
                                          from `smc.swing_highs_lows` function.
            range_percent (float): The percentage of the overall price range to define
                                   the clustering threshold for liquidity.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - Liquidity (int): 1 if bullish liquidity (multiple lows), -1 if bearish liquidity (multiple highs), NaN otherwise.
                - Level (float): The average price level of the liquidity zone.
                - End (int): The index of the last swing point included in the liquidity zone.
                - Swept (int): The index of the candle that "swept" (broke through) the liquidity zone.
                               NaN if not swept.
        """
        # Create a copy of swing_highs_lows to avoid modifying the original during processing
        shl = swing_highs_lows.copy()
        n = len(ohlc)  # Total number of candles

        # Calculate the absolute price range based on the entire OHLC data
        overall_price_range = ohlc["high"].max() - ohlc["low"].min()
        # Define the liquidity threshold (pip_range) based on the percentage
        pip_range = overall_price_range * range_percent

        # Preconvert relevant columns from shl to numpy arrays for faster access
        shl_HL = shl[
            "highlow"
        ].values.copy()  # Use .copy() to allow in-place modification (marking used swings)
        shl_Level = shl["Level"].values.copy()

        # Preconvert OHLC data to numpy arrays
        ohlc_high = ohlc["high"].values
        ohlc_low = ohlc["low"].values

        # Initialize output arrays with NaN (float32) to represent no liquidity by default
        liquidity = np.full(n, np.nan, dtype=np.float32)
        liquidity_level = np.full(n, np.nan, dtype=np.float32)
        liquidity_end = np.full(n, np.nan, dtype=np.float32)
        liquidity_swept = np.full(n, np.nan, dtype=np.float32)

        # --- Process Bullish Liquidity (multiple lows) ---
        # Find indices of all swing lows (highlow == -1)
        bull_indices = np.nonzero(shl_HL == -1)[
            0
        ]  # Note: Bullish liquidity is identified by clustering of *lows*

        for i in bull_indices:
            # Skip if this swing point has already been grouped into a previous liquidity zone
            if shl_HL[i] != -1:  # Check the mutable copy
                continue

            low_level = shl_Level[i]  # Current swing low level
            # Define the range around the current swing low
            range_low_bound = low_level - pip_range
            range_high_bound = low_level + pip_range

            group_levels = [low_level]  # Start a new group with the current swing low
            group_end_index = (
                i  # Tracks the last index included in the current liquidity group
            )

            # Determine the 'swept' index for this potential liquidity zone:
            # This is the first candle *after* the current swing low that breaks below the liquidity range.
            candle_check_start_index = i + 1
            swept_index = np.nan  # Initialize to NaN
            if candle_check_start_index < n:  # Ensure there are candles to check
                # Condition: low of subsequent candles drops below the lower bound of the range
                cond = ohlc_low[candle_check_start_index:] <= range_low_bound
                if np.any(cond):
                    # Find the index of the first candle that meets the condition
                    swept_index = candle_check_start_index + int(np.argmax(cond))

            # Iterate over subsequent swing low candidates to form a liquidity group
            for j in bull_indices:
                if j <= i:  # Only consider swing lows after the current one
                    continue
                # If a sweep has occurred and the candidate swing low is beyond the sweep point, stop grouping
                if not np.isnan(swept_index) and j >= swept_index:
                    break

                # If the candidate swing low is within the defined liquidity range
                if shl_HL[j] == -1 and (
                    range_low_bound <= shl_Level[j] <= range_high_bound
                ):
                    group_levels.append(shl_Level[j])  # Add its level to the group
                    group_end_index = j  # Update the last index of the group
                    shl_HL[j] = (
                        0  # Mark this swing low as 'used' so it's not processed again
                    )

            # If more than one swing low was grouped, it's a valid bullish liquidity zone
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(
                    group_levels
                )  # Calculate average level of the zone
                liquidity[i] = 1  # Mark as bullish liquidity
                liquidity_level[i] = avg_level
                liquidity_end[i] = group_end_index
                liquidity_swept[i] = swept_index

        # --- Process Bearish Liquidity (multiple highs) ---
        # Find indices of all swing highs (highlow == 1)
        bear_indices = np.nonzero(shl_HL == 1)[
            0
        ]  # Note: Bearish liquidity is identified by clustering of *highs*

        for i in bear_indices:
            if shl_HL[i] != 1:
                continue

            high_level = shl_Level[i]
            range_low_bound = high_level - pip_range
            range_high_bound = high_level + pip_range

            group_levels = [high_level]
            group_end_index = i

            # Determine the 'swept' index for this potential liquidity zone:
            # This is the first candle *after* the current swing high that breaks above the liquidity range.
            candle_check_start_index = i + 1
            swept_index = np.nan
            if candle_check_start_index < n:
                # Condition: high of subsequent candles rises above the upper bound of the range
                cond = ohlc_high[candle_check_start_index:] >= range_high_bound
                if np.any(cond):
                    swept_index = candle_check_start_index + int(np.argmax(cond))

            # Iterate over subsequent swing high candidates to form a liquidity group
            for j in bear_indices:
                if j <= i:
                    continue
                if not np.isnan(swept_index) and j >= swept_index:
                    break

                if shl_HL[j] == 1 and (
                    range_low_bound <= shl_Level[j] <= range_high_bound
                ):
                    group_levels.append(shl_Level[j])
                    group_end_index = j
                    shl_HL[j] = 0  # Mark as 'used'

            # If more than one swing high was grouped, it's a valid bearish liquidity zone
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(group_levels)
                liquidity[i] = -1  # Mark as bearish liquidity
                liquidity_level[i] = avg_level
                liquidity_end[i] = group_end_index
                liquidity_swept[i] = swept_index

        # Convert numpy arrays to pandas Series, preserving the original DataFrame's index
        liq_series = pd.Series(liquidity, name="Liquidity", index=ohlc.index)
        level_series = pd.Series(liquidity_level, name="Level", index=ohlc.index)
        end_series = pd.Series(liquidity_end, name="End", index=ohlc.index)
        swept_series = pd.Series(liquidity_swept, name="Swept", index=ohlc.index)

        # Concatenate all Series into a single DataFrame and return
        return pd.concat([liq_series, level_series, end_series, swept_series], axis=1)

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        """
        Calculates the high and low of the *previous* higher timeframe candle
        and indicates if these levels have been broken by the current candle.

        Args:
            ohlc (DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
                              Must have a DatetimeIndex.
            time_frame (str): The higher timeframe to resample to (e.g., "15m", "1H", "4H", "1D", "1W", "1M").

        Returns:
            DataFrame: A DataFrame with the following columns:
                - Previoushigh (float): The high of the previous higher timeframe candle.
                - Previouslow (float): The low of the previous higher timeframe candle.
                - Brokenhigh (int): 1 if the previous high was broken (current high > Previoushigh), 0 otherwise.
                - Brokenlow (int): 1 if the previous low was broken (current low < Previouslow), 0 otherwise.
        """
        # Ensure the DataFrame index is a DatetimeIndex, which is crucial for resampling
        # The inputvalidator ensures 'ohlc' is a DataFrame, but not necessarily its index type.
        # This line ensures it's a DatetimeIndex.
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            ohlc.index = pd.to_datetime(ohlc.index)

        # Initialize output arrays with zeros
        previous_high = np.zeros(len(ohlc), dtype=np.float32)
        previous_low = np.zeros(len(ohlc), dtype=np.float32)
        broken_high = np.zeros(len(ohlc), dtype=np.int32)
        broken_low = np.zeros(len(ohlc), dtype=np.int32)

        # Resample the OHLC data to the specified higher timeframe
        resampled_ohlc = (
            ohlc.resample(time_frame)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )  # Drop any rows where resampling resulted in all NaNs

        currently_broken_high = (
            False  # Flag to track if the previous high is currently broken
        )
        currently_broken_low = (
            False  # Flag to track if the previous low is currently broken
        )
        last_broken_time_idx = None  # Tracks the index of the higher timeframe candle whose levels are being checked

        # Iterate through each candle in the original (e.g., 5-minute) DataFrame
        for i in range(len(ohlc)):
            # Find the index of the *previous* complete resampled candle relative to the current 5-min candle
            # `resampled_ohlc.index < ohlc.index[i]` finds all resampled bars that occurred before the current bar.
            # `[-2]` gets the second-to-last one, which corresponds to the *previous complete* higher timeframe bar.
            resampled_previous_indices = np.where(resampled_ohlc.index < ohlc.index[i])[
                0
            ]

            if len(resampled_previous_indices) < 2:
                # If there aren't at least two previous resampled bars (current incomplete + previous complete),
                # then we don't have a 'previous' high/low to reference.
                previous_high[i] = np.nan
                previous_low[i] = np.nan
                continue  # Move to the next 5-min candle

            # Get the index of the actual *previous complete* higher timeframe bar
            previous_resampled_bar_idx = resampled_previous_indices[-2]

            # Reset broken flags if we've moved to a new previous higher timeframe bar
            if last_broken_time_idx != previous_resampled_bar_idx:
                currently_broken_high = False
                currently_broken_low = False
                last_broken_time_idx = previous_resampled_bar_idx

            # Assign the high and low from the *previous complete* higher timeframe bar
            previous_high[i] = resampled_ohlc["high"].iloc[previous_resampled_bar_idx]
            previous_low[i] = resampled_ohlc["low"].iloc[previous_resampled_bar_idx]

            # Update broken flags:
            # If current high breaks above previous high OR it was already broken, set to True
            currently_broken_high = (
                ohlc["high"].iloc[i] > previous_high[i] or currently_broken_high
            )
            # If current low breaks below previous low OR it was already broken, set to True
            currently_broken_low = (
                ohlc["low"].iloc[i] < previous_low[i] or currently_broken_low
            )

            # Record 1 if broken, 0 if not
            broken_high[i] = 1 if currently_broken_high else 0
            broken_low[i] = 1 if currently_broken_low else 0

        # Convert numpy arrays to pandas Series, preserving the original DataFrame's index
        previous_high_series = pd.Series(
            previous_high, name="Previoushigh", index=ohlc.index
        )
        previous_low_series = pd.Series(
            previous_low, name="Previouslow", index=ohlc.index
        )
        broken_high_series = pd.Series(broken_high, name="Brokenhigh", index=ohlc.index)
        broken_low_series = pd.Series(broken_low, name="Brokenlow", index=ohlc.index)

        # Concatenate all Series into a single DataFrame and return
        return pd.concat(
            [
                previous_high_series,
                previous_low_series,
                broken_high_series,
                broken_low_series,
            ],
            axis=1,
        )

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def sessions(
        cls,
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> DataFrame:
        """
        Determines if candles fall within predefined or custom trading sessions
        and calculates the high/low of the active session.

        Args:
            ohlc (DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
                              Must have a DatetimeIndex.
            session (str): The name of the session to check
                           (e.g., "Sydney", "Tokyo", "London", "New York",
                           "Asian kill zone", "London open kill zone",
                           "New York kill zone", "london close kill zone", "Custom").
            start_time (str): Start time of the custom session in "HH:MM" format.
                              Required only if `session` is "Custom".
            end_time (str): End time of the custom session in "HH:MM" format.
                            Required only if `session` is "Custom".
            time_zone (str): The timezone of the input `ohlc` data. Can be "UTC+0", "GMT+0", etc.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - Active (int): 1 if the candle is within the session, 0 if not.
                - high (float): The highest price reached within the *current* active session up to that candle.
                - low (float): The lowest price reached within the *current* active session up to that candle.
        """
        # Validate input for custom session
        if session == "Custom" and (start_time == "" or end_time == ""):
            raise ValueError("Custom session requires a start and end time (HH:MM).")

        # Define default session times (UTC)
        default_sessions = {
            "Sydney": {"start": "21:00", "end": "06:00"},
            "Tokyo": {"start": "00:00", "end": "09:00"},
            "London": {"start": "07:00", "end": "16:00"},
            "New York": {"start": "13:00", "end": "22:00"},
            "Asian kill zone": {"start": "00:00", "end": "04:00"},
            "London open kill zone": {
                "start": "06:00",
                "end": "09:00",
            },  # Corrected from 6:00
            "New York kill zone": {"start": "11:00", "end": "14:00"},
            "london close kill zone": {"start": "14:00", "end": "16:00"},
            "Custom": {"start": start_time, "end": end_time},
        }

        # Ensure DataFrame index is DatetimeIndex
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            ohlc.index = pd.to_datetime(ohlc.index)

        # Localize and convert timezone if specified
        if time_zone != "UTC":
            # Replace common timezone aliases to a format pandas understands
            time_zone = time_zone.replace("GMT", "Etc/GMT").replace("UTC", "Etc/GMT")
            # Localize the index to the specified timezone, then convert to UTC
            ohlc.index = ohlc.index.tz_localize(
                time_zone, ambiguous="NaT", nonexistent="NaT"
            ).tz_convert("UTC")
            # Handle potential NaT values from timezone conversion
            ohlc = ohlc.dropna(subset=[ohlc.index.name])

        # Parse session start and end times from the dictionary
        session_start_time_str = default_sessions[session]["start"]
        session_end_time_str = default_sessions[session]["end"]

        # Convert times to datetime objects for comparison (date part doesn't matter, only time)
        session_start_dt = datetime.strptime(session_start_time_str, "%H:%M").time()
        session_end_dt = datetime.strptime(session_end_time_str, "%H:%M").time()

        # Initialize output arrays
        active = np.zeros(len(ohlc), dtype=np.int32)
        session_high = np.zeros(len(ohlc), dtype=np.float32)
        session_low = np.zeros(len(ohlc), dtype=np.float32)

        current_session_high = 0.0  # Tracks high within the current active session
        current_session_low = float(
            "inf"
        )  # Tracks low within the current active session
        is_session_active_prev = False  # Flag to detect session transitions

        for i in range(len(ohlc)):
            current_candle_time = ohlc.index[
                i
            ].time()  # Get only the time part of the current candle's timestamp

            # Determine if the current candle is within the session
            is_active_now = False
            if session_start_dt < session_end_dt:
                # Normal session (e.g., 07:00 to 16:00)
                is_active_now = (
                    session_start_dt <= current_candle_time <= session_end_dt
                )
            else:
                # Overnight session (e.g., 21:00 to 06:00, crosses midnight)
                is_active_now = (current_candle_time >= session_start_dt) or (
                    current_candle_time <= session_end_dt
                )

            active[i] = 1 if is_active_now else 0

            # Handle session high/low tracking
            if is_active_now:
                if not is_session_active_prev:  # If session just started
                    current_session_high = ohlc["high"].iloc[i]
                    current_session_low = ohlc["low"].iloc[i]
                else:  # If session is ongoing
                    current_session_high = max(
                        current_session_high, ohlc["high"].iloc[i]
                    )
                    current_session_low = min(current_session_low, ohlc["low"].iloc[i])

                session_high[i] = current_session_high
                session_low[i] = current_session_low
            else:
                # If not active, reset session high/low for next session or keep as 0/inf
                current_session_high = 0.0
                current_session_low = float("inf")
                session_high[i] = (
                    np.nan
                )  # Or 0, depending on desired output for inactive periods
                session_low[i] = (
                    np.nan
                )  # Or 0, depending on desired output for inactive periods

            is_session_active_prev = is_active_now  # Update flag for next iteration

        # Convert numpy arrays to pandas Series, preserving the original DataFrame's index
        active_series = pd.Series(active, name="Active", index=ohlc.index)
        high_series = pd.Series(session_high, name="high", index=ohlc.index)
        low_series = pd.Series(session_low, name="low", index=ohlc.index)

        # Concatenate all Series into a single DataFrame and return
        return pd.concat([active_series, high_series, low_series], axis=1)

    @classmethod
    @inputvalidator(input_="ohlc")  # Apply decorator directly
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> DataFrame:
        """
        Calculates price retracements from identified swing highs and lows.
        A retracement is a temporary reversal in the direction of the price.

        Args:
            ohlc (DataFrame): DataFrame with 'high' and 'low' columns.
            swing_highs_lows (DataFrame): DataFrame containing 'highlow' and 'Level'
                                          from `smc.swing_highs_lows` function.

        Returns:
            DataFrame: A DataFrame with the following columns:
                - Direction (int): 1 if bullish retracement (from a low), -1 if bearish retracement (from a high), 0 otherwise.
                - CurrentRetracement% (float): The current percentage retracement from the last valid swing point.
                - DeepestRetracement% (float): The deepest percentage retracement reached so far within the current swing.
        """
        # Create a copy to avoid modifying the original DataFrame
        swing_highs_lows = swing_highs_lows.copy()

        # Initialize arrays for results
        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)

        current_top = 0.0  # Tracks the level of the last swing high
        current_bottom = 0.0  # Tracks the level of the last swing low
        current_trend_direction = 0  # 1 for bullish, -1 for bearish, 0 for undefined

        for i in range(len(ohlc)):
            # Update current swing high/low and trend direction if a new swing point is found
            if swing_highs_lows["highlow"].iloc[i] == 1:  # Found a swing high
                current_trend_direction = 1
                current_top = swing_highs_lows["Level"].iloc[i]
                # Reset deepest retracement when a new swing high is established
                deepest_retracement[i] = 0.0
            elif swing_highs_lows["highlow"].iloc[i] == -1:  # Found a swing low
                current_trend_direction = -1
                current_bottom = swing_highs_lows["Level"].iloc[i]
                # Reset deepest retracement when a new swing low is established
                deepest_retracement[i] = 0.0
            else:
                # If no new swing, maintain the previous direction
                current_trend_direction = direction[i - 1] if i > 0 else 0

            direction[i] = current_trend_direction  # Store the determined direction

            # Calculate retracement based on the current trend direction
            if (
                current_trend_direction == 1
            ):  # Bullish trend (price moving up, retracing down from a low)
                # Retracement from the bottom to the current low, relative to the swing range
                if current_top > current_bottom:  # Avoid division by zero
                    current_retracement[i] = round(
                        100
                        - (
                            (
                                (ohlc["low"].iloc[i] - current_bottom)
                                / (current_top - current_bottom)
                            )
                            * 100
                        ),
                        1,
                    )
                else:
                    current_retracement[i] = np.nan  # Invalid range

                # Update deepest retracement for the current bullish swing
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[
                            i - 1
                        ]  # Continue deepest from previous bar if still in same swing
                        if i > 0 and direction[i - 1] == 1
                        else 0.0  # Reset if new swing or initial state
                    ),
                    current_retracement[i],  # Compare with current retracement
                )
            elif (
                current_trend_direction == -1
            ):  # Bearish trend (price moving down, retracing up from a high)
                # Retracement from the top to the current high, relative to the swing range
                if current_bottom < current_top:  # Avoid division by zero
                    current_retracement[i] = round(
                        100
                        - (
                            (ohlc["high"].iloc[i] - current_top)
                            / (current_bottom - current_top)
                        )
                        * 100,
                        1,
                    )
                else:
                    current_retracement[i] = np.nan  # Invalid range

                # Update deepest retracement for the current bearish swing
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[
                            i - 1
                        ]  # Continue deepest from previous bar if still in same swing
                        if i > 0 and direction[i - 1] == -1
                        else 0.0  # Reset if new swing or initial state
                    ),
                    current_retracement[i],  # Compare with current retracement
                )
            else:  # Undefined direction
                current_retracement[i] = np.nan
                deepest_retracement[i] = np.nan

        # Shift the arrays by 1 to align with common indicator practice (current bar's retracement is for previous swing)
        # This means the retracement value at index `i` reflects the retracement of the swing that ended at `i-1`.
        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        # Remove the first few retracements as they might be calculated incorrectly due to insufficient initial data
        # This loop clears the first 3 retracement segments that occur after a direction change.
        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):  # Avoid index out of bounds
                break
            if direction[i] != direction[i + 1]:  # Detect a change in trend direction
                remove_first_count += 1

            # Clear data for the current bar if it's part of the initial "unreliable" segments
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0

            if remove_first_count == 3:  # Stop after clearing 3 segments
                # Also clear the next bar, as it's the start of the 4th segment
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        # Replace 0s (initial/cleared values) with NaN for cleaner output
        direction = np.where(direction != 0, direction, np.nan)
        current_retracement = np.where(
            current_retracement != 0, current_retracement, np.nan
        )
        deepest_retracement = np.where(
            deepest_retracement != 0, deepest_retracement, np.nan
        )

        # Convert numpy arrays to pandas Series, preserving the original DataFrame's index
        direction_series = pd.Series(direction, name="Direction", index=ohlc.index)
        current_retracement_series = pd.Series(
            current_retracement, name="CurrentRetracement%", index=ohlc.index
        )
        deepest_retracement_series = pd.Series(
            deepest_retracement, name="DeepestRetracement%", index=ohlc.index
        )

        # Concatenate all Series into a single DataFrame and return
        return pd.concat(
            [direction_series, current_retracement_series, deepest_retracement_series],
            axis=1,
        )
