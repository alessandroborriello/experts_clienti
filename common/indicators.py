"""
Indicatori tecnici - formule standard (EMA, Wilder RSI/ATR/ADX, Bollinger).

Nota sulla parità numerica con MetaTrader: queste sono le formule standard
(le stesse usate internamente da MT4/MT5), ma la fase di "innesco" (i primi
`period` valori, prima che la media si stabilizzi) può differire per
qualche millesimo dal valore esatto del terminale, perché MT5 inizializza
la media di Wilder con una SMA dei primi `period` valori mentre qui si usa
`ewm(adjust=False)` che inizializza col primo valore stesso. Con almeno
~5x il periodo più lungo di storico (qui si richiedono 400-500 barre, i
periodi più lunghi sono 96/200) la differenza diventa trascurabile, ma non
è stata validata byte-a-byte contro un terminale MT5 reale: prima di
fidarsi ciecamente per trading reale, confronta un po' di barre fianco a
fianco con i Print() dell'EA.

Nota sulle bande di Bollinger: l'EA MQL5 crea l'indicatore con
`bands_shift=2` (iBands(..., 20, 2, 2, PRICE_CLOSE) - periodo 20,
bands_shift 2, deviazione 2.0). L'effetto esatto di bands_shift sui valori
letti via CopyBuffer con shift 0 non è stato verificato contro un
terminale live: qui calcoliamo le bande "normali" (nessuno shift). Se le
bande risultano importanti per le decisioni di trading, verifica questo
punto prima di considerare il port equivalente all'EA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average, stessa formula ricorsiva standard di MT5."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Media di Wilder: alpha = 1/period (usata da ATR, RSI, ADX in MT4/MT5)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (smoothing di Wilder, come iATR in MT5)."""
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, period)


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index (smoothing di Wilder, come iRSI in MT5)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 e avg_gain > 0 -> RSI 100 (nessuna perdita, solo guadagni)
    out = out.where(avg_loss != 0.0, 100.0)
    # avg_loss == 0 e avg_gain == 0 -> nessun movimento, RSI neutro 50
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out


def bollinger_bands(close: pd.Series, period: int, deviation: float):
    """Bande di Bollinger "normali" (nessuno shift). Vedi nota di modulo sul
    bands_shift=2 usato dall'EA MQL5, non replicato qui."""
    mid = sma(close, period)
    std = close.rolling(window=period).std(ddof=0)   # deviazione standard di popolazione
    upper = mid + deviation * std
    lower = mid - deviation * std
    return upper, mid, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average Directional Index (Wilder), come iADX in MT5 (solo linea principale)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(high, low, close)
    atr_w = _wilder_smooth(tr, period)

    plus_dm_smooth = _wilder_smooth(plus_dm, period)
    minus_dm_smooth = _wilder_smooth(minus_dm, period)

    plus_di = 100.0 * (plus_dm_smooth / atr_w.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / atr_w.replace(0.0, np.nan))

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.fillna(0.0)

    return _wilder_smooth(dx, period)
