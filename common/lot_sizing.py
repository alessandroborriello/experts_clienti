"""
Calcolo del lottaggio da usare per aprire un'operazione.

Tre modalità, selezionabili per ogni strategia/cliente:

  "fixed"         - lotto fisso (equivalente a lot_size dell'EA quando
                     DynamicLotSize=false).
  "dynamic_risk"  - stesso calcolo di CalcDynamicLots() nell'EA: lotto
                     derivato da saldo conto, % di rischio e distanza
                     dello stop loss. Richiede dati live del broker
                     (saldo, tick_value, tick_size), passati dal motore.
  "profile_table" - interroga il Risk Profile Service (basso/medio/alto
                     -> lottaggio per simbolo) creato a parte. Vedi
                     risk_profile_service nel progetto gemello.

Nessuna di queste funzioni parla con MetaTrader5: ricevono i dati di
mercato/conto già letti dal motore, così restano testabili senza una
connessione live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class DynamicRiskInputs:
    """Dati di mercato/conto necessari per il calcolo "dynamic_risk"
    (equivalenti a quelli che l'EA legge da AccountInfoDouble/SymbolInfoDouble)."""
    balance: float
    max_risk_pct: float
    point: float
    tick_value: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float


def fixed_lot(lot_size: float) -> float:
    return lot_size


def dynamic_risk_lot(open_price: float, sl_price: float, inputs: DynamicRiskInputs,
                      fallback_lot: float) -> float:
    """Porting esatto di CalcDynamicLots() in MetodoATS_AIEvol_27.mq5."""
    if sl_price <= 0.0:
        return fallback_lot
    if inputs.tick_size <= 0.0 or inputs.point <= 0.0:
        return fallback_lot

    risk_money = inputs.balance * inputs.max_risk_pct / 100.0
    sl_points = abs(open_price - sl_price) / inputs.point
    value_per_point = (inputs.tick_value / inputs.tick_size * inputs.point) if inputs.tick_size > 0 else 0.0

    if sl_points <= 0.0 or value_per_point <= 0.0:
        return fallback_lot

    lots = risk_money / (sl_points * value_per_point)

    if inputs.volume_step > 0.0:
        lots = math.floor(lots / inputs.volume_step) * inputs.volume_step

    lots = max(lots, inputs.volume_min)
    lots = min(lots, inputs.volume_max)
    return lots


class LotSizeServiceError(RuntimeError):
    pass


def profile_table_lot(base_url: str, profile: str, symbol: str,
                       client_key: Optional[str] = None,
                       fallback_lot: Optional[float] = None,
                       timeout: float = 5.0) -> float:
    """Interroga GET /lot-size del Risk Profile Service.

    Solleva LotSizeServiceError se il servizio non risponde e non è stato
    passato un fallback_lot locale (meglio bloccare l'apertura con un
    errore chiaro che indovinare un lottaggio).
    """
    params = {"profile": profile, "symbol": symbol}
    if client_key:
        params["key"] = client_key

    try:
        resp = requests.get(f"{base_url.rstrip('/')}/lot-size", params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # rete, timeout, JSON invalido, HTTP error...
        if fallback_lot is not None:
            return fallback_lot
        raise LotSizeServiceError(f"Risk Profile Service non raggiungibile: {exc}") from exc

    if data.get("status") == "ok" and data.get("lot_size") is not None:
        return float(data["lot_size"])

    if fallback_lot is not None:
        return fallback_lot
    raise LotSizeServiceError(
        f"Nessun lottaggio configurato per profile={profile} symbol={symbol}: {data}"
    )
