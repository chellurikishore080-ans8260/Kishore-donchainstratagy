"""
Dukascopy historical data loader (free, years of 1-minute data) -> 15m and 30m candles.

Dukascopy publishes one LZMA-compressed file of 1-minute BID candles per instrument per UTC day:
    https://datafeed.dukascopy.com/datafeed/<SYMBOL>/<YYYY>/<MM 0-based>/<DD>/BID_candles_min_1.bi5
This module downloads those day files in parallel, decodes them, builds 15m and 30m candles and
keeps them in data/<SYMBOL>_15m.csv and data/<SYMBOL>_30m.csv (UTC timestamps). The next run only
downloads the days that are missing, so the cache grows instead of being rebuilt.

Command line (e.g. from GitHub Actions):
    python dukascopy.py --months 36
"""
import argparse
import lzma
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
TIMEFRAMES = {"15m": 15, "30m": 30}

# preferred = price divisor Dukascopy normally uses; lo/hi = sanity range of the real price.
# If the preferred divisor gives prices outside lo..hi, other divisors are tried automatically.
SYMBOLS = {
    "XAUUSD": dict(label="Gold", divisor=1000, lo=300, hi=10000),
    "USA500IDXUSD": dict(label="US500", divisor=1000, lo=500, hi=20000),
}
_DIVISORS = (1000, 100, 10, 10000, 100000, 1)

_DTYPE = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("h", ">u4"), ("l", ">u4"), ("v", ">f4")])


# ----------------------------------------------------------------------------------------------
# Decoding / download
# ----------------------------------------------------------------------------------------------
def decode_candles(raw: bytes, day: date) -> pd.DataFrame:
    """Decode one day file. Prices are returned UNSCALED (integers as floats); see apply_divisor()."""
    if not raw:
        return pd.DataFrame()
    data = lzma.decompress(raw)
    n = len(data) // _DTYPE.itemsize
    if n == 0:
        return pd.DataFrame()
    a = np.frombuffer(data[: n * _DTYPE.itemsize], dtype=_DTYPE)
    midnight = pd.Timestamp(day, tz="UTC")
    idx = midnight + pd.to_timedelta(a["t"].astype("int64"), unit="s")
    return pd.DataFrame({
        "open": a["o"].astype("float64"), "high": a["h"].astype("float64"),
        "low": a["l"].astype("float64"), "close": a["c"].astype("float64"),
        "volume": a["v"].astype("float64"),
    }, index=idx)


def apply_divisor(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Turn raw integer prices into real prices, picking the divisor that gives a plausible price."""
    med = float(df["close"].median())
    for div in (spec["divisor"],) + tuple(d for d in _DIVISORS if d != spec["divisor"]):
        if spec["lo"] <= med / div <= spec["hi"]:
            out = df.copy()
            out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]] / div
            return out
    raise RuntimeError(f"Dukascopy prices look wrong (median raw close {med:g}); no divisor gives "
                       f"{spec['lo']}-{spec['hi']}")


def _url(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"


def _fetch_day_bytes(symbol: str, day: date, retries: int = 3) -> bytes:
    """Raw bytes of one day file; b"" when Dukascopy has no file for that day (holiday / weekend)."""
    import requests
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(_url(symbol, day), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{symbol} {day}: {type(last_err).__name__}: {last_err}")


def _days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() != 5:               # no Saturday files
            yield d
        d += timedelta(days=1)


def fetch_minutes(symbol: str, start: date, end: date, workers: int = 8, progress=None) -> pd.DataFrame:
    """Download 1-minute candles (unscaled prices) for start..end inclusive."""
    days = list(_days(start, end))
    if not days:
        return pd.DataFrame()
    done = [0]

    def one(d):
        try:
            raw = _fetch_day_bytes(symbol, d)
            res = (decode_candles(raw, d), None)
        except Exception as e:  # noqa: BLE001
            res = (pd.DataFrame(), e)
        done[0] += 1
        if progress:
            progress(done[0] / len(days))
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, days))
    errors = [e for _, e in results if e is not None]
    frames = [f for f, _ in results if not f.empty]
    if errors and len(errors) > len(days) / 2:
        raise RuntimeError(f"Dukascopy download failed for {len(errors)} of {len(days)} days. {errors[0]}")
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def resample(df1m: pd.DataFrame, mins: int) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df1m.resample(f"{mins}min", closed="left", label="left").agg(agg)
    return out.dropna(subset=["open", "close"])


# ----------------------------------------------------------------------------------------------
# Cache (data/<SYMBOL>_<tf>.csv, UTC) and incremental refresh
# ----------------------------------------------------------------------------------------------
def _path(symbol: str, tf: str, data_dir: Path) -> Path:
    return Path(data_dir) / f"{symbol}_{tf}.csv"


def read_cache(symbol: str, tf: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    p = _path(symbol, tf, data_dir)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _write_cache(df: pd.DataFrame, symbol: str, tf: str, data_dir: Path):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "time"
    out.round(5).to_csv(_path(symbol, tf, data_dir))


def refresh(symbol: str, months: int = 12, data_dir: Path = DATA_DIR, workers: int = 8,
            progress=None, today: date = None) -> dict:
    """Make sure the cache covers the last `months` months, downloading only what is missing.
    Returns {"15m": df, "30m": df} (UTC index, open/high/low/close/volume), trimmed to `months`."""
    if symbol not in SYMBOLS:
        raise ValueError(f"unknown Dukascopy symbol {symbol}; known: {', '.join(SYMBOLS)}")
    spec = SYMBOLS[symbol]
    end = today or pd.Timestamp.now(tz="UTC").date()
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()

    cached = {tf: read_cache(symbol, tf, data_dir) for tf in TIMEFRAMES}
    have = all(not c.empty for c in cached.values())
    ranges, keep_before = [], None
    if not have:
        cached = {tf: pd.DataFrame() for tf in TIMEFRAMES}
        ranges.append((start, end))
    else:
        first, last = cached["15m"].index[0].date(), cached["15m"].index[-1].date()
        if start < first - timedelta(days=7):
            ranges.append((start, first - timedelta(days=1)))
        ranges.append((max(last, start), end))             # re-fetch the last cached day (it may be partial)
        keep_before = pd.Timestamp(max(last, start), tz="UTC")

    frames = []
    total = sum((b - a).days + 1 for a, b in ranges)
    base = 0
    for a, b in ranges:
        n = (b - a).days + 1
        cb = (lambda f, base=base, n=n: progress((base + f * n) / total)) if progress else None
        m = fetch_minutes(symbol, a, b, workers, cb)
        if not m.empty:
            frames.append(m)
        base += n

    if frames:
        df1m = apply_divisor(pd.concat(frames).sort_index(), spec)
        df1m = df1m[~df1m.index.duplicated(keep="last")]
        for tf, mins in TIMEFRAMES.items():
            new = resample(df1m, mins)
            old = cached[tf]
            if keep_before is not None and not old.empty:
                old = old[old.index < keep_before]
            merged = pd.concat([old, new]) if not old.empty else new
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            cached[tf] = merged
            _write_cache(merged, symbol, tf, data_dir)
    elif not have:
        raise RuntimeError(f"No Dukascopy data returned for {symbol} ({start} to {end}). "
                           f"Check the symbol name or your internet access.")

    cutoff = pd.Timestamp(start, tz="UTC")
    return {tf: df[df.index >= cutoff] for tf, df in cached.items()}


def main():
    ap = argparse.ArgumentParser(description="Download Dukascopy data and save 15m/30m CSVs in data/")
    ap.add_argument("--months", type=int, default=36, help="how many months of history to keep (default 36)")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS), help="Dukascopy symbols")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for s in a.symbols:
        print(f"{s}: downloading up to {a.months} months ...", flush=True)
        res = refresh(s, a.months, Path(a.data_dir), a.workers)
        for tf, df in res.items():
            print(f"  {tf}: {len(df)} candles  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
