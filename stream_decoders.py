# stream_decoders.py
from __future__ import annotations

from decimal import Decimal, getcontext
import numpy as np
import pandas as pd

__all__ = [
    "dec_last_digit_stream",
    "dec_times1000_mod2",
    "METHODS",
]


def dec_last_digit_stream(s: pd.Series) -> pd.Series:
    """
    'Siste siffer i en stream':
      - Konverterer verdier robust til streng (via Decimal for stabilitet)
      - Ignorerer tegn som ikke er [0-9]
      - Tar siste siffer og returnerer paritet (partall=0, oddetall=1)
    """
    getcontext().prec = 50  # robust presisjon ved formatering

    def last_digit_parity(x):
        if pd.isna(x):
            return 0
        try:
            d = Decimal(str(x))
            txt = format(d, "f")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if not digits:
                return 0
            return int(digits[-1]) % 2
        except Exception:
            # Fallback: float-format med fast presisjon
            try:
                txt = f"{float(x):.15f}"
                digits = "".join(ch for ch in txt if ch.isdigit())
                if not digits:
                    return 0
                return int(digits[-1]) % 2
            except Exception:
                return 0

    return s.apply(last_digit_parity).astype(int)


def dec_times1000_mod2(s: pd.Series) -> pd.Series:
    """
    Gang med 1000, ta floor, og sjekk paritet:
      - partall -> 0
      - oddetall -> 1
    """
    x = np.floor(s * 1000.0)
    # bruk abs før cast for å unngå negative mod-kanttilfeller
    return (np.abs(x).astype(np.int64) % 2).astype(int)


# Praktisk samling for grid-søk i hovedskriptet
METHODS = {
    "last_digit_stream": dec_last_digit_stream,
    "times1000_mod2": dec_times1000_mod2,
}
