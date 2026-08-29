from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentMetadata:
    symbol: str
    pip_size: float
    price_decimals: int

    @property
    def pip_scale(self) -> float:
        return 1.0 / self.pip_size


DEFAULT_INSTRUMENT = InstrumentMetadata(symbol="DEFAULT", pip_size=0.0001, price_decimals=5)


def get_instrument_metadata(symbol: str) -> InstrumentMetadata:
    normalized = symbol.upper().replace("/", "")
    if "JPY" in normalized:
        return InstrumentMetadata(symbol=symbol, pip_size=0.01, price_decimals=3)
    if normalized.startswith(("XAU", "GOLD")):
        return InstrumentMetadata(symbol=symbol, pip_size=0.01, price_decimals=2)
    if normalized.startswith(("US30", "NAS", "SPX", "GER", "UK100")):
        return InstrumentMetadata(symbol=symbol, pip_size=0.1, price_decimals=1)
    return InstrumentMetadata(
        symbol=symbol,
        pip_size=DEFAULT_INSTRUMENT.pip_size,
        price_decimals=DEFAULT_INSTRUMENT.price_decimals,
    )
