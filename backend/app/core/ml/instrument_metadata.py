import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InstrumentMetadata:
    symbol: str
    pip_size: float
    price_decimals: int

    @property
    def pip_scale(self) -> float:
        return 1.0 / self.pip_size


DEFAULT_INSTRUMENT = InstrumentMetadata(symbol="DEFAULT", pip_size=0.01, price_decimals=2)

KNOWN_EQUITY_ETFS = (
    "GLD", "SLV", "GDX", "USO", "UNG", "DBA", "DBC",
    "SPY", "QQQ", "IWM", "DIA", "TLT", "EEM", "EFA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLC", "XRT", "SMH",
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX", "COIN", "MSTR"
)


def get_instrument_metadata(symbol: str) -> InstrumentMetadata:
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if "JPY" in normalized:
        return InstrumentMetadata(symbol=symbol, pip_size=0.01, price_decimals=3)
    if any(k in normalized for k in ("BTC", "BITO")):
        return InstrumentMetadata(symbol=symbol, pip_size=1.0, price_decimals=2)
    if "ETH" in normalized:
        return InstrumentMetadata(symbol=symbol, pip_size=0.1, price_decimals=2)
    if any(k in normalized for k in ("US30", "NAS", "SPX", "GER", "UK100", "NDX")):
        return InstrumentMetadata(symbol=symbol, pip_size=0.1, price_decimals=1)
    if any(k in normalized for k in KNOWN_EQUITY_ETFS) or len(normalized) <= 5:
        # Standard US Equity / ETF: 1 cent = 1 pip ($0.01)
        return InstrumentMetadata(symbol=symbol, pip_size=0.01, price_decimals=2)

    logger.warning("[InstrumentMetadata] Unmapped symbol '%s' — falling back to Equity default (pip_size=0.01)", symbol)
    return InstrumentMetadata(
        symbol=symbol,
        pip_size=DEFAULT_INSTRUMENT.pip_size,
        price_decimals=DEFAULT_INSTRUMENT.price_decimals,
    )
