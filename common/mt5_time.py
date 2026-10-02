"""
Regole orarie (ora italiana) condivise da tutte le strategie.

Porting delle funzioni GetItalianTime() / IsAllowedTime() / CloseFridayTrades()
/ CloseTradesIfThursday22() di MetodoATS_AIEvol_27.mq5.

Convenzione day_of_week: usiamo la stessa convenzione di MQL5
(0=domenica, 1=lunedi, ..., 6=sabato), NON quella di Python
(datetime.weekday() e' 0=lunedi..6=domenica), per restare confrontabili
riga per riga con l'EA originale. Usa mql5_day_of_week() per convertire.

Differenza voluta rispetto all'EA: GetItalianTime() in MQL5 usa un calcolo
approssimato dell'ora legale (mese/giorno fissi invece dell'ultima domenica
di marzo/ottobre). Qui usiamo zoneinfo("Europe/Rome"), che e' corretto al
minuto in ogni periodo dell'anno. Per qualche giorno all'anno a cavallo del
cambio ora legale/solare, questo motore e l'EA MQL5 possono quindi differire
di un'ora sui controlli orario-dipendenti: tienilo a mente se confronti i
due side-by-side durante un periodo di validazione.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ITALY_TZ = ZoneInfo("Europe/Rome")


def mql5_day_of_week(dt: datetime) -> int:
    """datetime -> giorno della settimana in convenzione MQL5 (0=domenica..6=sabato)."""
    return dt.isoweekday() % 7


def to_italian_time(utc_dt: datetime) -> datetime:
    """Converte un datetime (deve essere timezone-aware, tipicamente UTC) in ora italiana."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(ITALY_TZ)


@dataclass(frozen=True)
class ItalianMoment:
    """Istante espresso in ora italiana, con i campi usati dalle regole di sessione."""
    dt: datetime
    day_of_week: int   # convenzione MQL5: 0=domenica..6=sabato
    hour: int
    minute: int

    @classmethod
    def from_utc(cls, utc_dt: datetime) -> "ItalianMoment":
        local = to_italian_time(utc_dt)
        return cls(dt=local, day_of_week=mql5_day_of_week(local), hour=local.hour, minute=local.minute)

    @property
    def minutes_of_day(self) -> int:
        return self.hour * 60 + self.minute


def is_allowed_time(moment: ItalianMoment) -> bool:
    """Porting di IsAllowedTime(): orario consentito per APRIRE nuove operazioni."""
    day, hour = moment.day_of_week, moment.hour

    if day == 0 or day >= 5:
        return False          # domenica, venerdi, sabato
    if day == 1 and hour < 9:
        return False          # lunedi prima delle 09:00
    if day == 4 and hour >= 21:
        return False          # giovedi dopo le 21:00
    return True


def should_close_friday(moment: ItalianMoment, close_hour: int, close_minute: int,
                         enabled: bool = True) -> bool:
    """Porting di CloseFridayTrades(): True se e' ora di chiudere tutto per il weekend."""
    if not enabled:
        return False
    if moment.day_of_week != 5:
        return False
    return moment.minutes_of_day >= close_hour * 60 + close_minute


def should_close_thursday(moment: ItalianMoment, close_hour: int, close_minute: int,
                           enabled: bool = False) -> bool:
    """Porting di CloseTradesIfThursday22(): disattivata di default, come nell'EA."""
    if not enabled:
        return False
    if moment.day_of_week != 4:
        return False
    return moment.minutes_of_day >= close_hour * 60 + close_minute
