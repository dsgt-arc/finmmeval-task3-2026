from enum import Enum


class Signal(str, Enum):
    """Signal type"""

    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

    def __str__(self) -> str:
        return self.value


class SignalNumerical(str, Enum):
    """Signal type"""

    BULLISH = +1
    BEARISH = -1
    NEUTRAL = 0

    def __str__(self) -> int:
        return self.value
