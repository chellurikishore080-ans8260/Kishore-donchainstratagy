"""
Donchian Breakout Scanner + Backtester
Instruments: Gold, Silver, US500  |  Timeframes: 5m, 15m, 30m, 1h, 4h

Rules (from the plan):
  LONG : bullish candle closes above previous N-candle Donchian upper (entry candle excluded)
         + EMA fast > EMA slow + EMA slow rising
  SHORT: bearish candle closes below previous N-candle Donchian lower (entry candle excluded)
         + EMA fast < EMA slow + EMA slow falling
  SL   : breakout candle low (long) / high (short)
  Risk : 0.5% of equity per trade, skip trade if SL distance > 1% of entry, one position at a time
  TP   : 2R | 3R | 50% @1R + 50% @2R | 50% @1R + 50% @3R

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
    "Silver (XAGUSD)": {"yahoo": "SI=F", "twelve": "XAG/USD"},
    "US500": {"yahoo": "ES=F", "twelve": "SPX"},
}

TIMEFRAMES = {
    "5m": {"mins": 5, "yahoo": "5m", "period": "60d", "twelve": "5min"},
    "15m": {"mins": 15, "yahoo": "15m", "period": "60d", "twelve": "15min"},
    "30m": {"mins": 30, "yahoo": "30m", "period": "60d", "twelve": "30min"},
    "1h": {"mins": 60, "yahoo": "1h", "period": "730d", "twelve": "1h"},
    "4h": {"mins": 240, "yahoo": "1h", "period": "730d", "twelve": "4h", "resample": "4h"},
}

EMA_PRESETS = {
    "Original 50/100": (50, 100),
    "Medium 30/60": (30, 60),
    "Fast 20/50": (20, 50),
    "Fastest 10/30": (10, 30),
}

TP_MODES = {
    "2R": (None, 2.0),
    "3R": (None, 3.0),
    "1R+2R": (1.0, 2.0),   # 50% at 1R, 50% at 2R
    "1R+3R": (1.0, 3.0),   # 50% at 1R, 50% at 3R
}

DISPLAY_TZ = "Asia/Kolkata"


@dataclass(frozen=True)
class Params:
    don_len: int = 20
    ema_fast: int = 50
    ema_slow: int = 100
    slope_bars: int = 1          # EMA slow must be above its value this many bars ago
    risk_pct: float = 0.5
    max_sl_pct: float = 1.0
    tp_mode: str = "2R"
    entry_mode: str = "close"    # "close" = enter at breakout candle close, "next_open" = next candle open
    be_after_tp1: bool = False   # move SL to entry after the 1R partial
    cost_pct: float = 0.0        # round-turn cost as % of price (spread + commission + slippage)
    capital: float = 10000.0
    allow_long: bool = True
    allow_short: bool = True


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, resample: str = None) -> pd.DataFrame:
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
    if resample:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        df = df.resample(resample).agg(agg).dropna(subset=need)
    df.index = df.index.tz_convert(DISPLAY_TZ)
    df.index.name = "time"
    return df


def _yahoo(symbol: str, tf: str) -> pd.DataFrame:
    import yfinance as yf
    cfg = TIMEFRAMES[tf]
    raw = yf.download(symbol, period=cfg["period"], interval=cfg["yahoo"],
                      auto_adjust=False, progress=False, threads=False)
    return _finalize(raw, cfg.get("resample"))


def _twelve(symbol: str, tf: str, api_key: str) -> pd.DataFrame:
    import requests
    cfg = TIMEFRAMES[tf]
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params=dict(symbol=symbol, interval=cfg["twelve"], outputsize=5000,
                    apikey=api_key, timezone="UTC", order="ASC"),
        timeout=30,
    )
    j = r.json()
    if "values" not in j:
        raise RuntimeError(j.get("message", "Twelve Data returned no values"))
    df = pd.DataFrame(j["values"]).set_index("datetime")
    df.index = pd.to_datetime(df.index)
    return _finalize(df)


def _csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((cols[k] for k in ["time", "datetime", "date", "timestamp", "gmt time",
                                   "local time", "date/time"] if k in cols), df.columns[0])
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, errors="coerce")
    out = df.drop(columns=[tcol]).copy()
    out.index = idx
    out = out[out.index.notna()]
    return _finalize(out)


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
    def load_twelve(symbol, tf, api_key):
        return _twelve(symbol, tf, api_key)
else:  # pragma: no cover
    load_yahoo, load_twelve = _yahoo, _twelve


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


def _simulate(h, l, c, j0, side, entry, sl, risk, p: Params):
    """Walk forward from bar j0. Returns (R_before_costs, exit_idx, reason). SL wins ties inside one bar."""
    n = len(c)
    part_at, final_r = TP_MODES[p.tp_mode]
    hs, ls = h[j0:], l[j0:]
    if side == 1:
        sl_hit = ls <= sl
        tpf_hit = hs >= entry + risk * final_r
        tp1_hit = hs >= entry + risk * part_at if part_at else None
    else:
        sl_hit = hs >= sl
        tpf_hit = ls <= entry - risk * final_r
        tp1_hit = ls <= entry - risk * part_at if part_at else None

    f_sl, f_tp = _first(sl_hit), _first(tpf_hit)

    def eod(rem=1.0):
        return rem * (c[-1] - entry) * side / risk

    if part_at is None:                                   # ---- full exit modes
        if np.isfinite(f_sl) and f_sl <= f_tp:
            return -1.0, j0 + int(f_sl), "SL"
        if np.isfinite(f_tp):
            return final_r, j0 + int(f_tp), "TP"
        return eod(), n - 1, "EOD"

    f_t1 = _first(tp1_hit)                                # ---- partial modes
    if np.isfinite(f_sl) and f_sl <= f_t1:
        return -1.0, j0 + int(f_sl), "SL"
    if not np.isfinite(f_t1):
        return eod(), n - 1, "EOD"
    t1 = int(f_t1)
    sl2 = entry if p.be_after_tp1 else sl
    # second leg: TP can hit on the same bar as TP1 (SL did not touch that bar), SL only checked after it
    tp_rel = _first(tpf_hit[t1:])
    tp_abs = t1 + tp_rel if np.isfinite(tp_rel) else np.inf
    if side == 1:
        sl2_hit = ls[t1 + 1:] <= sl2
    else:
        sl2_hit = hs[t1 + 1:] >= sl2
    s2_rel = _first(sl2_hit)
    s2_abs = t1 + 1 + s2_rel if np.isfinite(s2_rel) else np.inf
    if tp_abs < s2_abs:                                   # TP strictly earlier than SL2 (ties go to SL)
        r2, idx, why = final_r, j0 + int(tp_abs), "TP"
    elif np.isfinite(s2_abs):
        r2 = (sl2 - entry) * side / risk
        idx, why = j0 + int(s2_abs), ("BE" if p.be_after_tp1 else "SL2")
    else:
        r2, idx, why = eod(), n - 1, "EOD"
    return 0.5 * part_at + 0.5 * r2, idx, why


def backtest(df: pd.DataFrame, p: Params):
    """Returns (trades_df, indicators_df, skipped_count)."""
    d = add_indicators(df, p)
    o, h, l, c = (d[k].values for k in ["open", "high", "low", "close"])
    long_s, short_s = d["long_sig"].values, d["short_sig"].values
    idx = d.index
    n = len(d)
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
        r, exit_i, why = _simulate(h, l, c, i + 1, side, entry, sl, risk, p)
        r_net = r - (p.cost_pct / 100 * entry / risk)
        part_at, final_r = TP_MODES[p.tp_mode]
        rows.append(dict(
            entry_time=idx[i] if p.entry_mode == "close" else idx[i + 1],
            exit_time=idx[exit_i], side="LONG" if side == 1 else "SHORT",
            entry=entry, sl=sl, sl_pct=risk / entry * 100,
            tp1=(entry + side * risk * part_at) if part_at else np.nan,
            tp=entry + side * risk * final_r,
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
    trend = ("Bullish (EMA fast>slow, slow rising)" if last.ema_f > last.ema_s and d["ema_s"].iloc[-1] > d["ema_s"].iloc[-1 - p.slope_bars]
             else "Bearish (EMA fast<slow, slow falling)" if last.ema_f < last.ema_s and d["ema_s"].iloc[-1] < d["ema_s"].iloc[-1 - p.slope_bars]
             else "Mixed / no trend")
    out = dict(Candle=d.index[-1].strftime("%d-%b %H:%M"), Close=round(last.close, 3),
               Trend=trend, **{"Don upper": round(last.don_up, 3), "Don lower": round(last.don_lo, 3)})
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
        **{"Signal candle": d.index[i].strftime("%d-%b %H:%M"), "Bars ago": n - 1 - i,
           "Entry": round(entry, 3), "SL": round(sl, 3), "SL %": round(sl_pct, 3),
           "TP 1R": round(entry + side * risk, 3), "TP 2R": round(entry + side * risk * 2, 3),
           "TP 3R": round(entry + side * risk * 3, 3),
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
    fig.add_trace(go.Scatter(x=x.index, y=x.ema_f, name="EMA fast", line=dict(width=1.2, color="#ffa726")))
    fig.add_trace(go.Scatter(x=x.index, y=x.ema_s, name="EMA slow", line=dict(width=1.2, color="#42a5f5")))
    if not trades.empty:
        t = trades[trades["entry_time"] >= x.index[0]]
        for side, sym, col in [("LONG", "triangle-up", "#00e676"), ("SHORT", "triangle-down", "#ff5252")]:
            s = t[t.side == side]
            fig.add_trace(go.Scatter(x=s.entry_time, y=s.entry, mode="markers", name=f"{side} entry",
                                     marker=dict(symbol=sym, size=11, color=col)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h"))
    return fig


def main():
    st.set_page_config(page_title="Donchian Scanner + Backtest", layout="wide")
    st.title("Donchian Breakout — Scanner + Backtest")

    # ---------------- sidebar ----------------
    sb = st.sidebar
    source = sb.selectbox("Data source", ["Yahoo Finance", "Twelve Data", "CSV upload"])
    api_key = ""
    if source == "Twelve Data":
        api_key = sb.text_input("Twelve Data API key", type="password",
                                value=st.secrets.get("TWELVE_DATA_KEY", "") if hasattr(st, "secrets") else "")
    insts = sb.multiselect("Instruments", list(INSTRUMENTS), default=list(INSTRUMENTS))
    tfs = sb.multiselect("Timeframes", list(TIMEFRAMES), default=["5m", "15m"])
    csv_files = {}
    if source == "CSV upload":
        sb.caption("Upload OHLC CSV (time, open, high, low, close) per instrument. Same file is used for every selected timeframe.")
        for name in INSTRUMENTS:
            f = sb.file_uploader(f"{name} CSV", type=["csv", "txt"], key=f"csv_{name}")
            if f is not None:
                csv_files[name] = f.getvalue()

    sb.markdown("---")
    preset = sb.selectbox("EMA preset", list(EMA_PRESETS) + ["Custom"], index=2)
    if preset == "Custom":
        ef = sb.number_input("EMA fast", 2, 500, 20)
        es = sb.number_input("EMA slow", 3, 1000, 50)
    else:
        ef, es = EMA_PRESETS[preset]
        sb.caption(f"EMA fast {ef} / slow {es}")
    don = sb.number_input("Donchian length", 5, 200, 20)
    slope = sb.number_input("EMA-slow rising/falling check (bars)", 1, 20, 1)
    sb.markdown("---")
    risk = sb.number_input("Risk per trade %", 0.05, 5.0, 0.5, 0.05)
    max_sl = sb.number_input("Skip if SL distance > % of entry", 0.05, 10.0, 1.0, 0.05)
    entry_mode = sb.radio("Entry price", ["close", "next_open"], format_func=lambda x: "Breakout candle close" if x == "close" else "Next candle open")
    be = sb.checkbox("Move SL to breakeven after 1R partial", value=False)
    cost = sb.number_input("Round-turn cost % of price (spread+comm+slippage)", 0.0, 1.0, 0.0, 0.005, format="%.3f")
    capital = sb.number_input("Starting capital", 100.0, 10_000_000.0, 10000.0, 100.0)
    direction = sb.radio("Direction", ["Both", "Long only", "Short only"], horizontal=True)

    base = Params(don_len=int(don), ema_fast=int(ef), ema_slow=int(es), slope_bars=int(slope),
                  risk_pct=float(risk), max_sl_pct=float(max_sl), entry_mode=entry_mode,
                  be_after_tp1=be, cost_pct=float(cost), capital=float(capital),
                  allow_long=direction != "Short only", allow_short=direction != "Long only")

    def get(inst, tf, live_trim=True):
        try:
            if source == "Yahoo Finance":
                df, live = load_yahoo(INSTRUMENTS[inst]["yahoo"], tf), True
            elif source == "Twelve Data":
                if not api_key:
                    return None, "Enter Twelve Data API key in the sidebar"
                df, live = load_twelve(INSTRUMENTS[inst]["twelve"], tf, api_key), True
            else:
                if inst not in csv_files:
                    return None, "Upload a CSV for this instrument"
                df, live = _csv(csv_files[inst]), False
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        if df is None or df.empty:
            return None, "No data returned"
        if live and live_trim:
            df = drop_unclosed(df, tf)
        if len(df) < max(base.don_len, base.ema_slow) + 20:
            return None, f"Only {len(df)} candles — not enough"
        return df, None

    tab_scan, tab_bt, tab_sweep, tab_rules = st.tabs(["Scanner", "Backtest", "EMA / TP sweep", "Rules"])

    # ---------------- scanner ----------------
    with tab_scan:
        c1, c2, c3 = st.columns([1, 1, 2])
        lookback = c1.number_input("Show signals from last N candles", 1, 20, 3)
        auto = c2.checkbox("Auto-refresh (60s)")
        if c3.button("Refresh now"):
            st.cache_data.clear()
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
            live = tbl[tbl["Signal"].isin(["LONG", "SHORT"])]
            if live.empty:
                st.info("No signal on the selected candles right now.")
        for e in errs:
            st.warning(e)
        if source == "Yahoo Finance":
            st.caption("Yahoo gives only ~60 days of 5m/15m data. US500 uses ES=F futures, Gold GC=F, Silver SI=F "
                       "(futures prices differ slightly from CFD prices). Use Twelve Data or CSV for longer history.")
        if auto:
            time.sleep(60)
            st.cache_data.clear()
            st.rerun()

    # ---------------- backtest ----------------
    with tab_bt:
        if not insts or not tfs:
            st.info("Pick instruments and timeframes in the sidebar.")
        else:
            b1, b2, b3 = st.columns(3)
            inst = b1.selectbox("Instrument", insts, key="bt_inst")
            tf = b2.selectbox("Timeframe", tfs, key="bt_tf")
            mode = b3.selectbox("TP mode to inspect", list(TP_MODES), key="bt_tp")
            df, err = get(inst, tf)
            if err:
                st.warning(err)
            else:
                st.caption(f"{len(df)} candles  |  {df.index[0]:%d-%b-%Y %H:%M} → {df.index[-1]:%d-%b-%Y %H:%M} IST  |  "
                           f"EMA {base.ema_fast}/{base.ema_slow}, Donchian {base.don_len}")
                comp, store = [], {}
                for m in TP_MODES:
                    tr, dd, sk = backtest(df, replace(base, tp_mode=m))
                    store[m] = (tr, dd)
                    comp.append(dict(TP=m, **summarize(tr, replace(base, tp_mode=m), sk)))
                st.subheader("All 4 TP experiments")
                st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

                tr, dd = store[mode]
                st.subheader(f"Detail — {mode}")
                if tr.empty:
                    st.info("No trades with these settings.")
                else:
                    eq = equity_curve(tr, replace(base, tp_mode=mode))
                    st.line_chart(eq, height=240)
                    st.plotly_chart(make_chart(dd, tr), use_container_width=True)
                    show = tr.drop(columns=["entry_i", "exit_i"]).round(4)
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    st.download_button("Download trades CSV", show.to_csv(index=False).encode(),
                                       file_name=f"donchian_{inst.split()[0]}_{tf}_{mode}.csv")

    # ---------------- sweep ----------------
    with tab_sweep:
        st.write("Runs every combination of EMA preset × Donchian length × TP mode on the selected instruments/timeframes.")
        s1, s2, s3 = st.columns(3)
        presets = s1.multiselect("EMA presets", list(EMA_PRESETS), default=list(EMA_PRESETS))
        dons = s2.multiselect("Donchian lengths", [5, 10, 14, 20, 30, 50], default=[10, 20])
        min_tr = s3.number_input("Min trades to show", 0, 1000, 20)
        if st.button("Run sweep", type="primary"):
            out, prog = [], st.progress(0.0)
            combos = [(i, t) for i in insts for t in tfs]
            for k, (inst, tf) in enumerate(combos):
                df, err = get(inst, tf)
                if err:
                    st.warning(f"{inst} {tf}: {err}")
                else:
                    for pn in presets:
                        for dl in dons:
                            for m in TP_MODES:
                                pp = replace(base, ema_fast=EMA_PRESETS[pn][0], ema_slow=EMA_PRESETS[pn][1],
                                             don_len=int(dl), tp_mode=m)
                                tr, _, sk = backtest(df, pp)
                                out.append(dict(Instrument=inst, TF=tf, EMA=pn, Donchian=dl, TP=m, **summarize(tr, pp, sk)))
                prog.progress((k + 1) / len(combos))
            if out:
                res = pd.DataFrame(out)
                res = res[res["Trades"] >= min_tr].sort_values("Return %", ascending=False)
                st.dataframe(res, use_container_width=True, hide_index=True)
                st.download_button("Download sweep CSV", res.to_csv(index=False).encode(), file_name="donchian_sweep.csv")
                st.caption("Best row in a sweep is partly luck. Prefer settings that work across several instruments/timeframes "
                           "with enough trades, and confirm on data the sweep did not see.")

    # ---------------- rules ----------------
    with tab_rules:
        st.markdown(f"""
**LONG** — bullish candle closes above the previous **{base.don_len}**-candle Donchian upper (entry candle excluded),
EMA **{base.ema_fast}** > EMA **{base.ema_slow}**, EMA {base.ema_slow} rising.

**SHORT** — bearish candle closes below the previous {base.don_len}-candle Donchian lower (entry candle excluded),
EMA {base.ema_fast} < EMA {base.ema_slow}, EMA {base.ema_slow} falling.

**SL** — breakout candle low (long) / high (short). **Risk** {base.risk_pct}% per trade, skip if SL > {base.max_sl_pct}% of entry, one position at a time.

**TP tests** — 2R, 3R, 50% @ 1R + 50% @ 2R, 50% @ 1R + 50% @ 3R.

Backtest notes: if SL and TP are both inside one candle, SL is assumed first. Position size = risk % of current equity ÷ SL distance,
so results are in R and equity compounds. Cost input is subtracted from every trade in R.
""")


if __name__ == "__main__" and st is not None:
    main()
