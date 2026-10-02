# expert_clienti

Porting in Python degli algoritmi che oggi vivono come Expert Advisor
MQL5, per farli girare dal motore Python (Opzione A, deciso nelle
discussioni precedenti: il motore si collega al terminale MetaTrader già
aperto e loggato sul broker tramite il package `MetaTrader5`, legge
prezzi/spread/saldo reali da lì ed esegue gli ordini — niente dati
inventati, sempre e solo quello che manda il broker).

Questo è il sottoinsieme da distribuire ai clienti tramite la console: va
pensato come il contenuto del futuro repo Git privato "vetrina" di cui
si era parlato, non come il repo di sviluppo completo.

## Stato attuale

**Fatto** (testato, senza bisogno di una connessione MT5 live):
- `common/mt5_time.py` — ora italiana (zoneinfo, più precisa dell'euristica
  dell'EA) e regole di sessione/chiusura (Friday/Thursday close, orario
  consentito).
- `common/indicators.py` — EMA, RSI/ATR/ADX (Wilder), Bollinger Bands.
- `common/lot_sizing.py` — 3 modalità di lottaggio: fisso, dinamico per
  rischio (porting di `CalcDynamicLots`), tabella per profilo (chiama il
  microservizio `risk_profile_service` creato a parte).
- `common/risk_rules.py` — porting di `IsDailyLossHit`/`IsSemaphoreTriggered`.
- `strategies/base.py` — il contratto comune che ogni strategia deve
  rispettare (vedi sotto).
- `strategies/metodo_ats_aievol_27.py` — porting completo di
  **MetodoATS_AIEvol_27.mq5**: stesse feature ML (87 campi + account),
  stesso endpoint, stessa matematica di TP/SL/breakeven/semaforo/limite
  giornaliero/chiusure.
- `tests/` — 60+ test (unittest) che verificano tutto quanto sopra.

**NON fatto** (è il prossimo pezzo, "il motore"):
- Nessun codice qui si collega davvero a MetaTrader5. Le strategie
  ricevono dati già pronti e restituiscono *decisioni* (vedi
  `strategies/base.py`): un ciclo esterno (il motore) dovrà leggere i dati
  reali da `MetaTrader5`, chiamarle, ed eseguire quello che decidono con
  `mt5.order_send()`/`mt5.positions_get()` ecc. Quel ciclo non esiste
  ancora in questo progetto.
- Nessuna console/GUI. Nessun collegamento al catalogo Git dei
  clienti/aggiornamenti.

## Come far girare i test

```
cd expert_clienti
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

(Il package `MetaTrader5` non è richiesto per i test: nessun modulo qui
lo importa, perché nessuno parla davvero con MetaTrader — vedi sopra.)

## Il contratto (`strategies/base.py`)

Principio: **le strategie non parlano mai con MetaTrader5**. Ricevono dati
già pronti e restituiscono decisioni:

- `compute_features(closed_bars)` → dict di feature tecniche. `closed_bars`
  è un DataFrame con SOLO barre chiuse (niente barra in formazione),
  ordinate per tempo crescente: `.iloc[-1]` è l'ultima barra chiusa.
- `build_ml_payload(features, account, symbol)` → il JSON da mandare al
  servizio ML.
- `parse_ml_response(raw_json)` → estrae tipo di trade (BUY/SELL/niente) e
  confidence dalla risposta del servizio ML.
- `decide_signal(prediction, ctx)` → un `Signal` (direzione, TP, SL,
  lottaggio) oppure `None`. `ctx` (`DecisionContext`) porta tutto quello
  che serve: prezzo di riferimento, se l'orario è consentito, se il
  semaforo/limite giornaliero sono scattati, se c'è già una posizione
  aperta.
- `manage_open_positions(symbol, positions, market, now_epoch, italian_moment)`
  → lista di `Action` (chiudi / modifica SL-TP) da applicare a posizioni
  già aperte.

Un motore che implementa questo ciclo per qualsiasi strategia che rispetti
il contratto, senza codice su misura per ciascuna.

## Fedeltà al port — cosa sapere prima di fidarsi ciecamente

Letto tutto il file `.mq5` riga per riga, ma alcune cose NON sono (e non
possono essere, da qui) state verificate byte-a-byte contro un terminale
MetaTrader live:

1. **Indicatori**: le formule in `common/indicators.py` sono quelle
   standard (identiche nella definizione a quelle di MT4/MT5), ma la fase
   di innesco dei primi N valori usa un seed leggermente diverso da quello
   interno del terminale (si stabilizza comunque dopo pochi multipli del
   periodo — qui si usano 450+ barre, ampiamente a sufficienza per i
   periodi più lunghi usati, 96 e 200).
2. **Bollinger Bands**: l'EA crea l'indicatore con `bands_shift=2`
   (`iBands(...,20,2,2,...)`). L'effetto esatto di quel parametro sul
   valore letto via `CopyBuffer` con shift 0 non è stato verificato dal
   vivo: qui si calcolano le bande "normali" (nessuno shift).
3. **Ora legale**: l'EA calcola l'ora italiana con un'euristica
   approssimata (mese/giorno fissi); qui si usa `zoneinfo("Europe/Rome")`,
   corretto al minuto. Per pochi giorni l'anno, a cavallo del cambio
   ora, i due possono differire di un'ora sui controlli di sessione.
4. **3 feature sempre a zero**: `atr_22_growing_previous`,
   `atr_48_growing_previous`, `atr_96_growing_previous` sono dichiarate
   nell'EA ma mai calcolate (solo le varianti `_mean` lo sono) — quindi
   vengono sempre inviate come 0 al modello. Li ho preservati identici
   (0 fisso) per non disallinearmi in silenzio da un modello che è stato
   probabilmente addestrato con quei 3 campi sempre a zero.

**Prima di fidarsi per trading reale**: fai girare l'EA originale e questo
motore in parallelo (stesso simbolo, stesso istante) per un po' di barre,
loggando le feature calcolate da entrambi, e confronta. Se vedi
differenze sistematiche oltre quelle elencate sopra, è lì che vanno
cercate.
