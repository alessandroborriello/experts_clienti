"""
Scopre quali terminali MetaTrader 5 sono installati sul PC, SENZA
avviarli né collegarsi: legge solo i file che MetaTrader scrive da solo
nella sua cartella dati (origin.txt, config/common.ini).

Serve a rispondere alla domanda "su quale MetaTrader si aggancia il
motore?": prima di questo modulo, MT5Gateway.connect() veniva chiamato
senza argomenti, il che significa "il terminale già in esecuzione, o
quello di default del sistema" - ambiguo quando sul PC ce ne sono più di
uno (es. uno per MQL5, uno per MQL4, uno per un altro broker). Con
discover_terminals() si può invece scegliere ESPLICITAMENTE quale
terminal64.exe passare a MT5Gateway.connect(path=...).

Logica portata 1:1 da un progetto già in produzione
(machine-learning-percentuali/dashboard/server.py::discover_terminals),
non reinventata.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

TERMINALS_ROOT = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal")


@dataclass(frozen=True)
class TerminalInfo:
    id: str                    # nome della cartella dati (es. 8CD222C4...)
    name: str                  # nome della cartella di installazione
    exe: str                   # percorso completo di terminal64.exe
    login: Optional[str]       # account configurato (se mai collegato una volta)
    server: Optional[str]      # server/broker configurato
    running: bool              # True se il processo risulta già avviato


def _read_text(path: str) -> str:
    """Legge un file di testo di MetaTrader, che può essere UTF-16 (con
    BOM) o UTF-8 a seconda della versione del terminale che l'ha scritto."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return ""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def _running_executables() -> set[str]:
    try:
        import psutil
    except ImportError:
        return set()
    try:
        return {(p.info["exe"] or "").lower() for p in psutil.process_iter(["exe"]) if p.info.get("exe")}
    except Exception:
        return set()


def discover_terminals(terminals_root: str = TERMINALS_ROOT) -> list[TerminalInfo]:
    """Elenca i terminali MT5 installati, leggendo le cartelle dati sotto
    %APPDATA%\\MetaQuotes\\Terminal\\*. Non avvia né collega nulla."""
    running = _running_executables()

    found: list[TerminalInfo] = []
    for data_dir in sorted(glob.glob(os.path.join(terminals_root, "*"))):
        install = _read_text(os.path.join(data_dir, "origin.txt")).strip().lstrip("﻿")
        exe = os.path.join(install, "terminal64.exe")
        if not install or not os.path.isfile(exe):
            continue   # MT4, o installazione rimossa: terminal64.exe non c'è (serve l'API MT5)

        ini = _read_text(os.path.join(data_dir, "config", "common.ini"))
        login = re.search(r"^Login=(\S+)", ini, re.M)
        server = re.search(r"^Server=(.+)$", ini, re.M)

        found.append(TerminalInfo(
            id=os.path.basename(data_dir),
            name=os.path.basename(install.rstrip("\\/")),
            exe=exe,
            login=login.group(1) if login else None,
            server=server.group(1).strip() if server else None,
            running=exe.lower() in running,
        ))
    return found
