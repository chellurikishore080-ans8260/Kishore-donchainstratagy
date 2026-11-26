"""
Donchian Breakout Scanner + Backtester
Instruments: Gold, US500  |  Timeframes: 15m, 30m  |  EMA 50/100  |  TP: 1:2, 1:3

Rules:
  LONG : bullish candle closes above previous 20-candle Donchian upper (entry candle excluded)
         + EMA50 > EMA100 + EMA100 rising
  SHORT: bearish candle closes below previous 20-candle Donchian lower (entry candle excluded)
         + EMA50 < EMA100 + EMA100 falling
  SL   : breakout candle low (long) / high (short)
  Risk : 1% of equity per trade, skip trade if SL distance > 1% of entry, one position at a time
  TP   : full exit at 1:2 | 1:3

Run:  streamlit run app.py
"""
import io
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # lets the engine be imported/tested without streamlit
    st = None

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
INSTRUMENTS = {
    "Gold (XAUUSD)": {"yahoo": "GC=F", "twelve": "XAU/USD"},
    "US500": {"yahoo": "ES=F", "twelve": "SPX"},
}

TIMEFRAMES = {
    "15m": {"mins": 15, "yahoo": "15m", "period": "60d", "twelve": "15min"},
    "30m": {"mins": 30, "yahoo": "30m", "period": "60d", "twelve": "30min"},
}

# Take-profit as reward:risk, full exit
TP_RR = {"1:2": 2.0, "1:3": 3.0}

EMA_FAST, EMA_SLOW, DON_LEN, SLOPE_BARS = 50, 100, 20, 1   # fixed logic

DISPLAY_TZ = "Asia/Kolkata"


@dataclass(frozen=True)
class Params:
    don_len: int = DON_LEN
    ema_fast: int = EMA_FAST
    ema_slow: int = EMA_SLOW
    slope_bars: int = SLOPE_BARS
    risk_pct: float = 1.0
    max_sl_pct: float = 1.0
    tp_mode: str = "1:2"
    entry_mode: str = "close"    # "close" = enter at breakout candle close, "next_open" = next candle open
    cost_pct: float = 0.0        # round-turn cost as % of price (spread + commission + slippage)
    capital: float = 10000.0
    allow_long: bool = True
    allow_short: bool = True


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns/index: lower-case OHLC, tz-aware, sorted, no dupes, shown in IST."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    need = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()
    keep = need + (["volume"] if "volume" in df.columns else [])
    df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=need)
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert(DISPLAY_TZ)
    df.index.name = "time"
    return df


def _yahoo(symbol: str, tf: str) -> pd.DataFrame:
    import yfinance as yf
    cfg = TIMEFRAMES[tf]
    raw = yf.download(symbol, period=cfg["period"], interval=cfg["yahoo"],
                      auto_adjust=False, progress=False, threads=False)
    return _finalize(raw)


_last_td_call = [0.0]


def _td_throttle(min_gap: float = 7.5):
    """Twelve Data free plan allows 8 requests/min, so keep requests ~7.5s apart."""
    wait = min_gap - (time.time() - _last_td_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_td_call[0] = time.time()


def _twelve(symbol: str, tf: str, api_key: str, months: int = 1) -> pd.DataFrame:
    """Fetch history in 5000-candle pages, walking back from now until `months` are covered."""
    import requests
    cfg = TIMEFRAMES[tf]
    target = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months)
    frames, end, retries = [], None, 0
    for _ in range(60):
        params = dict(symbol=symbol, interval=cfg["twelve"], outputsize=5000,
                      apikey=api_key, timezone="UTC", order="DESC")
        if end:
            params["end_date"] = end
        _td_throttle()
        j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30).json()
        if "values" not in j:
            if j.get("code") == 429 and retries < 3:       # per-minute limit hit: wait and retry
                retries += 1
                time.sleep(60)
                continue
            if frames:                                      # no older data available: keep what we have
                break
            raise RuntimeError(j.get("message", "Twelve Data returned no values"))
        page = pd.DataFrame(j["values"])
        page["datetime"] = pd.to_datetime(page["datetime"])
        frames.append(page)
        oldest = page["datetime"].min()
        if len(page) < 5000 or oldest.tz_localize("UTC") <= target:
            break
        end = (oldest - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat(frames).drop_duplicates("datetime").set_index("datetime").sort_index()
    return _finalize(df)


def _csv(file_bytes: bytes) -> pd.DataFrame:
    """Read broker / Kaggle OHLC CSVs: with or without a header, single or separate date + time columns."""
    raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    names = {str(c).strip().lower().strip("<>"): c for c in raw.columns}
    if not all(k in names for k in ["open", "high", "low", "close"]):
        # headerless export (MT4 style): name the columns by how many there are
        raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", header=None)
        cols = {7: ["date", "time", "open", "high", "low", "close", "volume"],
                6: ["datetime", "open", "high", "low", "close", "volume"],
                5: ["datetime", "open", "high", "low", "close"]}.get(raw.shape[1])
        if cols is None:
            return pd.DataFrame()
        raw.columns = cols
        names = {c: c for c in cols}

    if "date" in names and "time" in names and names["date"] != names["time"]:
        t = raw[names["date"]].astype(str).str.strip() + " " + raw[names["time"]].astype(str).str.strip()
    else:
        tcol = next((names[k] for k in ["time", "datetime", "date", "timestamp", "gmt time",
                                        "local time", "date/time"] if k in names), raw.columns[0])
        t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        s = t.astype(str).str.strip()
        dayfirst = bool(s.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}").mean() > 0.9)   # 05.01.2020 = 5 Jan
        idx = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    out = raw[[names["open"], names["high"], names["low"], names["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    out.index = idx
    out = out[out.index.notna()]
    return _finalize(out)


def to_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Convert finer candles (1m, 5m, 15m...) to the requested timeframe; refuse to invent finer candles."""
    if df.empty:
        return df
    base = df.index.to_series().diff().dropna().median().total_seconds() / 60
    target = TIMEFRAMES[tf]["mins"]
    if base > target + 0.01:
        raise ValueError(f"your CSV has {base:g}-minute candles, so {tf} cannot be built from it")
    if base >= target - 0.01:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return df.resample(f"{target}min").agg(agg).dropna(subset=["open", "close"])


def drop_unclosed(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Remove the still-forming last candle (live sources only)."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz=df.index.tz)
    if now < df.index[-1] + pd.Timedelta(minutes=TIMEFRAMES[tf]["mins"]):
        return df.iloc[:-1]
    return df


if st is not None:
    @st.cache_data(ttl=300, show_spinner=False)
    def load_yahoo(symbol, tf):
        return _yahoo(symbol, tf)

    @st.cache_data(ttl=300, show_spinner=False)
    def load_twelve_recent(symbol, tf, api_key):          # scanner: latest ~5000 candles, refreshed often
        return _twelve(symbol, tf, api_key, 1)

    @st.cache_data(ttl=21600, show_spinner=False)
    def load_twelve_hist(symbol, tf, api_key, months):    # backtest: months of history, cached 6 hours
        return _twelve(symbol, tf, api_key, months)

    def clear_live_cache():
        load_yahoo.clear()
        load_twelve_recent.clear()
else:  # pragma: no cover
    load_yahoo = _yahoo
    load_twelve_recent = lambda s, t, k: _twelve(s, t, k, 1)          # noqa: E731
    load_twelve_hist = lambda s, t, k, m: _twelve(s, t, k, m)         # noqa: E731

    def clear_live_cache():
        pass


# ----------------------------------------------------------------------------------------------
# Strategy engine
# ----------------------------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["don_up"] = d["high"].rolling(p.don_len).max().shift(1)   # previous N candles, entry candle excluded
    d["don_lo"] = d["low"].rolling(p.don_len).min().shift(1)
    d["ema_f"] = d["close"].ewm(span=p.ema_fast, adjust=False).mean()
    d["ema_s"] = d["close"].ewm(span=p.ema_slow, adjust=False).mean()
    rising = d["ema_s"] > d["ema_s"].shift(p.slope_bars)
    falling = d["ema_s"] < d["ema_s"].shift(p.slope_bars)
    bull = d["close"] > d["open"]
    bear = d["close"] < d["open"]
    long_sig = bull & (d["close"] > d["don_up"]) & (d["ema_f"] > d["ema_s"]) & rising
    short_sig = bear & (d["close"] < d["don_lo"]) & (d["ema_f"] < d["ema_s"]) & falling
    warm = max(p.don_len, p.ema_slow) + p.slope_bars + 1      # let EMAs settle
    long_sig.iloc[:warm] = False
    short_sig.iloc[:warm] = False
    d["long_sig"] = long_sig & p.allow_long
    d["short_sig"] = short_sig & p.allow_short
    return d


def _first(mask: np.ndarray) -> float:
    if mask.size == 0:
        return np.inf
    k = int(mask.argmax())
    return float(k) if mask[k] else np.inf


def _simulate(h, l, c, j0, side, entry, sl, risk, rr):
    """Walk forward from bar j0, full exit at SL or rr*R. Returns (R, exit_idx, reason). SL wins ties inside one bar."""
    n = len(c)
    hs, ls = h[j0:], l[j0:]
    if side == 1:
        sl_hit = ls <= sl
        tp_hit = hs >= entry + risk * rr
    else:
        sl_hit = hs >= sl
        tp_hit = ls <= entry - risk * rr
    f_sl, f_tp = _first(sl_hit), _first(tp_hit)
    if np.isfinite(f_sl) and f_sl <= f_tp:
        return -1.0, j0 + int(f_sl), "SL"
    if np.isfinite(f_tp):
        return rr, j0 + int(f_tp), "TP"
    return (c[-1] - entry) * side / risk, n - 1, "EOD"      # still open at end of data


def backtest(df: pd.DataFrame, p: Params):
    """Returns (trades_df, indicators_df, skipped_count)."""
    d = add_indicators(df, p)
    o, h, l, c = (d[k].values for k in ["open", "high", "low", "close"])
    long_s, short_s = d["long_sig"].values, d["short_sig"].values
    idx = d.index
    n = len(d)
    rr = TP_RR[p.tp_mode]
    rows, skipped, last_exit = [], 0, -1
    for i in np.flatnonzero(long_s | short_s):
        if i <= last_exit or i + 1 >= n:
            continue                                      # one position at a time
        side = 1 if long_s[i] else -1
        entry = c[i] if p.entry_mode == "close" else o[i + 1]
        sl = l[i] if side == 1 else h[i]
        risk = (entry - sl) * side
        if risk <= 0:
            continue
        if risk / entry * 100 > p.max_sl_pct:
            skipped += 1
            continue
        r, exit_i, why = _simulate(h, l, c, i + 1, side, entry, sl, risk, rr)
        r_net = r - (p.cost_pct / 100 * entry / risk)
        rows.append(dict(
            entry_time=idx[i] if p.entry_mode == "close" else idx[i + 1],
            exit_time=idx[exit_i], side="LONG" if side == 1 else "SHORT",
            entry=entry, sl=sl, sl_pct=risk / entry * 100,
            tp=entry + side * risk * rr,
            exit_reason=why, R=r_net, bars=exit_i - i,
            entry_i=i, exit_i=exit_i,
        ))
        last_exit = exit_i
    return pd.DataFrame(rows), d, skipped


def summarize(trades: pd.DataFrame, p: Params, skipped: int = 0) -> dict:
    if trades.empty:
        return dict(Trades=0, **{"Win %": np.nan, "Total R": 0.0, "Avg R": np.nan, "PF": np.nan,
                                 "Max DD %": 0.0, "Return %": 0.0, "Final equity": p.capital,
                                 "Max loss streak": 0, "Skipped (SL>max)": skipped})
    r = trades["R"].values
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else (np.inf if len(wins) else np.nan)
    eq = p.capital * np.cumprod(1 + p.risk_pct / 100 * r)
    peak = np.maximum.accumulate(np.r_[p.capital, eq])[1:]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    return {
        "Trades": len(r), "Win %": round((r > 0).mean() * 100, 1), "Total R": round(r.sum(), 2),
        "Avg R": round(r.mean(), 3), "PF": round(pf, 2) if np.isfinite(pf) else pf,
        "Max DD %": round(((eq / peak) - 1).min() * 100, 2),
        "Return %": round((eq[-1] / p.capital - 1) * 100, 2), "Final equity": round(eq[-1], 2),
        "Max loss streak": best, "Skipped (SL>max)": skipped,
    }


def equity_curve(trades: pd.DataFrame, p: Params) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    eq = p.capital * np.cumprod(1 + p.risk_pct / 100 * trades["R"].values)
    return pd.Series(eq, index=trades["exit_time"].values)


def scan_one(df: pd.DataFrame, p: Params, lookback: int = 3) -> dict:
    """Latest-state scan on the last closed candle (+ any signal in the last `lookback` candles)."""
    d = add_indicators(df, p)
    n = len(d)
    last = d.iloc[-1]
    up = d["ema_s"].iloc[-1] > d["ema_s"].iloc[-1 - p.slope_bars]
    down = d["ema_s"].iloc[-1] < d["ema_s"].iloc[-1 - p.slope_bars]
    trend = ("Bullish (EMA50>100, 100 rising)" if last.ema_f > last.ema_s and up
             else "Bearish (EMA50<100, 100 falling)" if last.ema_f < last.ema_s and down
             else "Mixed / no trend")
    out = dict(Candle=d.index[-1].strftime("%d-%b %H:%M"), Close=round(float(last.close), 3),
               Trend=trend, **{"Don upper": round(float(last.don_up), 3), "Don lower": round(float(last.don_lo), 3)})
    sig = d["long_sig"].values | d["short_sig"].values
    recent = np.flatnonzero(sig[-lookback:])
    if len(recent) == 0:
        out.update(Signal="—")
        return out
    i = n - lookback + recent[-1]
    side = 1 if d["long_sig"].values[i] else -1
    entry = d["close"].values[i]
    sl = d["low"].values[i] if side == 1 else d["high"].values[i]
    risk = (entry - sl) * side
    sl_pct = risk / entry * 100 if risk > 0 else np.nan
    valid = np.isfinite(sl_pct) and sl_pct <= p.max_sl_pct
    out.update(
        Signal="LONG" if side == 1 else "SHORT",
        **{"Signal candle": d.index[i].strftime("%d-%b %H:%M"), "Bars ago": int(n - 1 - i),
           "Entry": round(float(entry), 3), "SL": round(float(sl), 3), "SL %": round(float(sl_pct), 3),
           "TP 1:2": round(float(entry + side * risk * 2), 3),
           "TP 1:3": round(float(entry + side * risk * 3), 3),
           "Status": "VALID" if valid else f"SKIP (SL>{p.max_sl_pct}%)"})
    return out


# ----------------------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------------------
def make_chart(d: pd.DataFrame, trades: pd.DataFrame, bars: int = 400):
    import plotly.graph_objects as go
    x = d.iloc[-bars:]
    fig = go.Figure(go.Candlestick(x=x.index, open=x.open, high=x.high, low=x.low, close=x.close, name="Price"))
    fig.add_trace(go.Scatter(x=x.index, y=x.don_up, name="Donchian up", line=dict(width=1, color="#26a69a")))
    fig.add_trace(go.Scatter(x=x.index, y=x.don_lo, name="Donchian low", line=dict(width=1, color="#ef5350")))
    fig.add_trace(go.Scatter(x=x.index, y=x.ema_f, name="EMA 50", line=dict(width=1.2, color="#ffa726")))
    fig.add_trace(go.Scatter(x=x.index, y=x.ema_s, name="EMA 100", line=dict(width=1.2, color="#42a5f5")))
    if not trades.empty:
        t = trades[trades["entry_time"] >= x.index[0]]
        for side, sym, col in [("LONG", "triangle-up", "#00e676"), ("SHORT", "triangle-down", "#ff5252")]:
            s = t[t.side == side]
            fig.add_trace(go.Scatter(x=s.entry_time, y=s.entry, mode="markers", name=f"{side} entry",
                                     marker=dict(symbol=sym, size=11, color=col)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h"))
    return fig


def _secret(name: str) -> str:
    """Read a Streamlit secret; returns "" when no secrets file is configured."""
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001  (StreamlitSecretNotFoundError when secrets.toml is missing)
        return ""


def main():
    st.set_page_config(page_title="Donchian Scanner + Backtest", layout="wide")
    st.title("Donchian Breakout — Scanner + Backtest")
    st.caption("Gold & US500  |  15m & 30m  |  EMA 50/100  |  TP 1:2 / 1:3")

    # ---------------- sidebar ----------------
    sb = st.sidebar
    source = sb.selectbox("Data source", ["Yahoo Finance", "Twelve Data", "CSV upload"])
    api_key = ""
    if source == "Twelve Data":
        api_key = sb.text_input("Twelve Data API key", type="password",
                                value=_secret("TWELVE_DATA_KEY"))
    months = 2
    if source == "Twelve Data":
        months = int(sb.number_input("Backtest history (months)", 1, 24, 6))
        sb.caption("Free plan allows 8 requests/min, so long history loads slowly the first time "
                   "(about 8s per 5000 candles), then it is cached for 6 hours.")
    insts = sb.multiselect("Instruments", list(INSTRUMENTS), default=list(INSTRUMENTS))
    tfs = sb.multiselect("Timeframes", list(TIMEFRAMES), default=list(TIMEFRAMES))
    csv_files = {}
    if source == "CSV upload":
        sb.caption("Upload one OHLC CSV per instrument (Kaggle / MT4 / MT5 exports work, with or without header). "
                   "Candles of 15m or finer are converted to the selected 15m / 30m automatically.")
        for name in INSTRUMENTS:
            f = sb.file_uploader(f"{name} CSV", type=["csv", "txt"], key=f"csv_{name}")
            if f is not None:
                csv_files[name] = f.getvalue()

    sb.markdown("---")
    risk = sb.number_input("Risk per trade %", 0.05, 5.0, 1.0, 0.05)
    max_sl = sb.number_input("Skip if SL distance > % of entry", 0.05, 10.0, 1.0, 0.05)
    entry_mode = sb.radio("Entry price", ["close", "next_open"],
                          format_func=lambda x: "Breakout candle close" if x == "close" else "Next candle open")
    cost = sb.number_input("Round-turn cost % of price (spread+comm+slippage)", 0.0, 1.0, 0.0, 0.005, format="%.3f")
    capital = sb.number_input("Starting capital", 100.0, 10_000_000.0, 10000.0, 100.0)
    direction = sb.radio("Direction", ["Both", "Long only", "Short only"], horizontal=True)

    base = Params(risk_pct=float(risk), max_sl_pct=float(max_sl), entry_mode=entry_mode,
                  cost_pct=float(cost), capital=float(capital),
                  allow_long=direction != "Short only", allow_short=direction != "Long only")

    def get(inst, tf, hist=False):
        try:
            if source == "Yahoo Finance":
                df, live = load_yahoo(INSTRUMENTS[inst]["yahoo"], tf), True
            elif source == "Twelve Data":
                if not api_key:
                    return None, "Enter Twelve Data API key in the sidebar"
                sym = INSTRUMENTS[inst]["twelve"]
                df = load_twelve_hist(sym, tf, api_key, months) if hist else load_twelve_recent(sym, tf, api_key)
                live = True
            else:
                if inst not in csv_files:
                    return None, "Upload a CSV for this instrument"
                df, live = to_timeframe(_csv(csv_files[inst]), tf), False
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        if df is None or df.empty:
            return None, "No data returned"
        if live:
            df = drop_unclosed(df, tf)
        if len(df) < max(base.don_len, base.ema_slow) + 20:
            return None, f"Only {len(df)} candles — not enough"
        return df, None

    tab_scan, tab_bt, tab_rules = st.tabs(["Scanner", "Backtest", "Rules"])

    # ---------------- scanner ----------------
    with tab_scan:
        c1, c2, c3 = st.columns([1, 1, 2])
        lookback = c1.number_input("Show signals from last N candles", 1, 20, 3)
        auto = c2.checkbox("Auto-refresh (60s)")
        if c3.button("Refresh now"):
            clear_live_cache()
        rows, errs = [], []
        for inst in insts:
            for tf in tfs:
                df, err = get(inst, tf)
                if err:
                    errs.append(f"{inst} {tf}: {err}")
                    continue
                rows.append(dict(Instrument=inst, TF=tf, **scan_one(df, base, int(lookback))))
        if rows:
            tbl = pd.DataFrame(rows)

            def colour(v):
                return ("background-color:#0d4d2b;color:white" if v == "LONG" else
                        "background-color:#6b1a1a;color:white" if v == "SHORT" else "")
            st.dataframe(tbl.style.map(colour, subset=["Signal"]), use_container_width=True, hide_index=True)
            if not tbl["Signal"].isin(["LONG", "SHORT"]).any():
                st.info("No signal on the selected candles right now.")
        for e in errs:
            st.warning(e)
        if source == "Yahoo Finance":
            st.caption("Yahoo gives only ~60 days of 15m/30m data. US500 uses ES=F futures and Gold uses GC=F "
                       "(futures prices differ slightly from CFD prices). Use Twelve Data or CSV for longer history.")
        if source == "CSV upload":
            st.caption("Scanner reads the last candle of your CSV, so it is only live if the file is up to date.")
        if auto:
            time.sleep(60)
            clear_live_cache()
            st.rerun()

    # ---------------- backtest ----------------
    with tab_bt:
        if not insts or not tfs:
            st.info("Pick instruments and timeframes in the sidebar.")
        else:
            if source == "Yahoo Finance":
                st.info("Yahoo only provides the last ~60 days of 15m/30m data. "
                        "Pick Twelve Data (set history in months) or upload a CSV for a longer backtest.")
            data, summary = {}, []
            with st.spinner("Loading history… first load of several months can take a few minutes on the free plan."):
                for inst in insts:
                    for tf in tfs:
                        df, err = get(inst, tf, hist=True)
                        if err:
                            st.warning(f"{inst} {tf}: {err}")
                            continue
                        data[(inst, tf)] = df
                        for m in TP_RR:
                            pp = replace(base, tp_mode=m)
                            tr, _, sk = backtest(df, pp)
                            summary.append(dict(Instrument=inst, TF=tf, TP=m, **summarize(tr, pp, sk)))
            if summary:
                st.subheader("Summary — all instruments × timeframes × TP")
                sm = pd.DataFrame(summary)
                st.dataframe(sm, use_container_width=True, hide_index=True)
                st.download_button("Download summary CSV", sm.to_csv(index=False).encode(),
                                   file_name="donchian_summary.csv")

                st.subheader("Detail")
                avail = list(data)
                b1, b2, b3 = st.columns(3)
                inst = b1.selectbox("Instrument", sorted({k[0] for k in avail}), key="bt_inst")
                tf = b2.selectbox("Timeframe", [k[1] for k in avail if k[0] == inst], key="bt_tf")
                mode = b3.selectbox("TP", list(TP_RR), index=0, key="bt_tp")
                df = data[(inst, tf)]
                st.caption(f"{len(df)} candles  |  {df.index[0]:%d-%b-%Y %H:%M} → {df.index[-1]:%d-%b-%Y %H:%M} IST")
                pp = replace(base, tp_mode=mode)
                tr, dd, sk = backtest(df, pp)
                if tr.empty:
                    st.info("No trades with these settings.")
                else:
                    st.line_chart(equity_curve(tr, pp), height=240)
                    st.plotly_chart(make_chart(dd, tr), use_container_width=True)
                    show = tr.drop(columns=["entry_i", "exit_i"]).round(4)
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    st.download_button("Download trades CSV", show.to_csv(index=False).encode(),
                                       file_name=f"donchian_{inst.split()[0]}_{tf}_{mode.replace(':', '-')}.csv")

    # ---------------- rules ----------------
    with tab_rules:
        st.markdown(f"""
**LONG** — bullish candle closes above the previous **{base.don_len}**-candle Donchian upper (entry candle excluded),
EMA **{base.ema_fast}** > EMA **{base.ema_slow}**, EMA {base.ema_slow} rising.

**SHORT** — bearish candle closes below the previous {base.don_len}-candle Donchian lower (entry candle excluded),
EMA {base.ema_fast} < EMA {base.ema_slow}, EMA {base.ema_slow} falling.

**SL** — breakout candle low (long) / high (short). **Risk** {base.risk_pct}% per trade, skip if SL > {base.max_sl_pct}% of entry, one position at a time.

**TP** — full exit at 1:2 or 1:3.

Backtest notes: if SL and TP are both inside one candle, SL is assumed first. Position size = risk % of current equity ÷ SL distance,
so results are in R and equity compounds. Cost input is subtracted from every trade in R.
""")


if __name__ == "__main__" and st is not None:
    main()
