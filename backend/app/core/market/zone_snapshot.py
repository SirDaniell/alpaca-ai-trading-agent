"""
zone_snapshot.py — Zone Snapshot Manager, Versioning & Hard Action Masking.

Architecture:
- HTF SNR Zones (M30, H1, H4, D1, W1) computed with zonal volume profiles.
- ZoneSnapshotManager stores timestamped, versioned zone snapshots (append-only).
- Active evaluation set is the union of valid snapshots with confluence weighting.
- Invalidation occurs strictly on confirmed candle closes past upper/lower bounds.
- HardActionMask enforces the no-chase entry rule and volume delta confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ZoneRecord:
    """Represents a single Support or Resistance zone with volume metrics."""
    zone_id: str
    timeframe: str  # "M30", "H1", "H4", "D1", "W1"
    price_level: float
    upper_bound: float
    lower_bound: float
    zone_type: str  # "support" or "resistance"
    total_volume: float = 0.0
    up_volume: float = 0.0
    down_volume: float = 0.0
    net_volume: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confluence_score: float = 1.0
    is_invalidated: bool = False

    @property
    def volume_delta_ratio(self) -> float:
        """Net volume ratio [-1.0, 1.0]."""
        if self.total_volume <= 0:
            return 0.0
        return float(np.clip(self.net_volume / self.total_volume, -1.0, 1.0))

    def contains_price(self, price: float, atr: float, buffer_mult: float = 0.5) -> bool:
        """Check if price falls within zone boundaries extended by ATR buffer."""
        buf = atr * buffer_mult
        return (self.lower_bound - buf) <= price <= (self.upper_bound + buf)


@dataclass
class ZoneSnapshot:
    """Immutable snapshot of zones computed at a specific candle close timestamp."""
    snapshot_id: str
    timestamp: datetime
    timeframe: str
    zones: List[ZoneRecord]


class ZoneSnapshotManager:
    """
    Manages rolling timestamped snapshots of HTF S&R zones.
    - Append-only rolling snapshot store (never overwrites).
    - Merges active valid snapshots into a unified confluence-weighted zone set.
    - Invalidates zones strictly on confirmed candle closes beyond zone bounds.
    """

    def __init__(self, max_snapshots: int = 20, atr_buffer_mult: float = 0.5):
        self.max_snapshots = max_snapshots
        self.atr_buffer_mult = atr_buffer_mult
        self.history: List[ZoneSnapshot] = []
        self._active_zones: Dict[str, ZoneRecord] = {}

    def add_snapshot(
        self,
        snapshot_id: str,
        timeframe: str,
        zones_raw: List[Tuple[int, float, List[Any], Dict[str, Any]]],
        timestamp: Optional[datetime] = None,
    ) -> ZoneSnapshot:
        """
        Incorporate raw SNR zones from support_resistance.py into snapshot history.

        raw zone format: (cluster_id, zone_price, cluster_levels, volume_data)
        """
        ts = timestamp or datetime.now(timezone.utc)
        records: List[ZoneRecord] = []

        for idx, z_price, cluster_levels, vol_data in zones_raw:
            # Determine zone type based on cluster levels
            types = [lvl[2] for lvl in cluster_levels if len(lvl) > 2]
            z_type = max(set(types), key=types.count) if types else "support"

            vol = vol_data or {}
            upper = vol.get("upper_bound", z_price * 1.004)
            lower = vol.get("lower_bound", z_price * 0.996)

            z_id = f"{timeframe}_{ts.strftime('%Y%m%d%H%M')}_{z_type}_{z_price:.4f}"
            rec = ZoneRecord(
                zone_id=z_id,
                timeframe=timeframe,
                price_level=float(z_price),
                upper_bound=float(upper),
                lower_bound=float(lower),
                zone_type=z_type,
                total_volume=float(vol.get("total_volume", 0.0)),
                up_volume=float(vol.get("up_volume", 0.0)),
                down_volume=float(vol.get("down_volume", 0.0)),
                net_volume=float(vol.get("net_volume", 0.0)),
                created_at=ts,
            )
            records.append(rec)
            self._active_zones[z_id] = rec

        snap = ZoneSnapshot(
            snapshot_id=snapshot_id,
            timestamp=ts,
            timeframe=timeframe,
            zones=records,
        )
        self.history.append(snap)
        if len(self.history) > self.max_snapshots:
            self.history.pop(0)

        self._recompute_confluence()
        return snap

    def update_invalidation(self, close_price: float, high_price: float, low_price: float) -> List[str]:
        """
        Invalidate active zones strictly on confirmed candle closes past bounds.
        - Support is invalidated if close_price < lower_bound.
        - Resistance is invalidated if close_price > upper_bound.

        Returns list of invalidated zone_ids.
        """
        invalidated = []
        for z_id, zone in list(self._active_zones.items()):
            if zone.is_invalidated:
                continue
            if zone.zone_type == "support" and close_price < zone.lower_bound:
                zone.is_invalidated = True
                invalidated.append(z_id)
            elif zone.zone_type == "resistance" and close_price > zone.upper_bound:
                zone.is_invalidated = True
                invalidated.append(z_id)

        # Remove invalidated zones from active store
        for z_id in invalidated:
            self._active_zones.pop(z_id, None)

        if invalidated:
            logger.info("[ZoneSnapshot] Invalidated %d zones on close_price=%.4f", len(invalidated), close_price)
        return invalidated

    def _recompute_confluence(self) -> None:
        """
        Compute confluence scores for active zones based on overlap across snapshots.
        Zones matching across multiple timeframes / historical snapshots gain higher scores.
        """
        active_list = list(self._active_zones.values())
        for z in active_list:
            score = 1.0
            for snap in self.history:
                for other in snap.zones:
                    if other.zone_id == z.zone_id or other.is_invalidated:
                        continue
                    # Overlap check
                    if abs(other.price_level - z.price_level) / z.price_level < 0.005:
                        score += 0.5
            z.confluence_score = round(score, 2)

    def get_active_zones(self, zone_type: Optional[str] = None) -> List[ZoneRecord]:
        """Return active valid (non-invalidated) zones sorted by price."""
        res = [z for z in self._active_zones.values() if not z.is_invalidated]
        if zone_type:
            res = [z for z in res if z.zone_type == zone_type]
        return sorted(res, key=lambda x: x.price_level)

    def get_nearest_zones(self, current_price: float) -> Tuple[Optional[ZoneRecord], Optional[ZoneRecord]]:
        """Return (nearest_support, nearest_resistance) relative to current price."""
        supports = self.get_active_zones(zone_type="support")
        resistances = self.get_active_zones(zone_type="resistance")

        supp_below = [s for s in supports if s.price_level <= current_price]
        nearest_supp = max(supp_below, key=lambda s: s.price_level) if supp_below else None

        res_above = [r for r in resistances if r.price_level >= current_price]
        nearest_res = min(res_above, key=lambda r: r.price_level) if res_above else None

        return nearest_supp, nearest_res


class HardActionMask:
    """
    Enforces no-chase entry discipline and volume delta confirmation.

    Action Index Mapping:
    - 0: WAIT
    - 1: BUY_CALL (Long Entry)
    - 2: BUY_PUT  (Short Entry)
    - 3: TAKE_PROFIT_HALF
    - 4: CLOSE_FLATTEN

    Rules:
    1. BUY_CALL is masked (0) unless price is within proximity of a valid Support zone
       AND volume delta confirms buying reaction (net_volume >= 0 or sell volume drying).
    2. BUY_PUT is masked (0) unless price is within proximity of a valid Resistance zone
       AND volume delta confirms selling reaction (net_volume <= 0 or buy volume drying).
    3. Exits/Adjustments (WAIT, TAKE_PROFIT_HALF, CLOSE_FLATTEN) are ALWAYS unmasked (1).
    """

    def __init__(self, proximity_atr_mult: float = 0.75, require_volume_confirm: bool = True):
        self.proximity_atr_mult = proximity_atr_mult
        self.require_volume_confirm = require_volume_confirm

    def get_action_mask(
        self,
        current_price: float,
        atr: float,
        zone_manager: ZoneSnapshotManager,
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        has_open_position: bool = False,
    ) -> np.ndarray:
        """
        Return binary mask array of shape (5,) where 1 = allowed, 0 = masked.
        Strict Single Trade Rule:
        - If has_open_position is True: New entries (BUY_CALL, BUY_PUT) are strictly MASKED OUT (0).
          Only position management actions (TAKE_PROFIT_HALF, CLOSE_FLATTEN, WAIT) are permitted.
        - If has_open_position is False: Position management actions (TAKE_PROFIT_HALF, CLOSE_FLATTEN) are MASKED OUT (0).
          New entries (BUY_CALL, BUY_PUT) are evaluated based on zone proximity and volume delta.
        """
        mask = np.array([1, 0, 0, 0, 0], dtype=np.int32)  # WAIT always allowed

        if has_open_position:
            mask[3] = 1  # TAKE_PROFIT_HALF
            mask[4] = 1  # CLOSE_FLATTEN
            # Single Running Trade Restriction: Cannot take new entries while a trade is active
            return mask

        nearest_supp, nearest_res = zone_manager.get_nearest_zones(current_price)
        prox_dist = max(atr * self.proximity_atr_mult, current_price * 0.003)

        # ── Check BUY_CALL (Long at Support) ──────────────────────────────────
        if nearest_supp and not nearest_supp.is_invalidated:
            dist_to_supp = abs(current_price - nearest_supp.price_level)
            if dist_to_supp <= prox_dist:
                vol_ok = True
                if self.require_volume_confirm:
                    # Require positive net volume or buy_volume >= sell_volume
                    vol_ok = (buy_volume >= sell_volume * 0.8) or (nearest_supp.net_volume >= 0)
                if vol_ok:
                    mask[1] = 1

        # ── Check BUY_PUT (Short at Resistance) ───────────────────────────────
        if nearest_res and not nearest_res.is_invalidated:
            dist_to_res = abs(current_price - nearest_res.price_level)
            if dist_to_res <= prox_dist:
                vol_ok = True
                if self.require_volume_confirm:
                    # Require negative net volume or sell_volume >= buy_volume
                    vol_ok = (sell_volume >= buy_volume * 0.8) or (nearest_res.net_volume <= 0)
                if vol_ok:
                    mask[2] = 1

        return mask

