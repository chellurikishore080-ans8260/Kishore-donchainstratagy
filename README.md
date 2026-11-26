# Donchian Breakout Strategy — Scanner + Backtest

Streamlit app for **Gold** and **US500** on **15m** and **30m** only, **EMA 50/100**.

## Rules
**LONG**
1. Donchian upper of the previous 20 candles (entry candle excluded)
2. Bullish candle closes above that upper band
3. EMA 50 > EMA 100
4. EMA 100 rising

**SHORT** — mirror image (bearish candle closes below Donchian lower, EMA 50 < EMA 100, EMA 100 falling).

**SL**: breakout candle low (long) / high (short)
**Risk**: 1% per trade, skip the trade if SL distance > 1% of entry, one position at a time
**TP**: full exit at **1:2** or **1:3** (no partials)

## Tabs
- **Scanner** — latest signal on each instrument/timeframe (entry, SL, SL %, TP 1:2 / 1:3, VALID / SKIP).
- **Backtest** — summary of Gold/US500 × 15m/30m × 3 TP ratios, then detail with equity curve, chart and trades CSV.

## Data sources
- **Yahoo Finance** (default): Gold `GC=F`, US500 `ES=F`. Only ~60 days of 15m/30m history.
- **Twelve Data**: months of intraday history. Set "Backtest history (months)" in the sidebar; the app pulls 5000-candle pages
  going back in time (free plan: 8 requests/min, so the first load of several months takes a few minutes, then it is cached for 6 hours).
  Add `TWELVE_DATA_KEY` in Streamlit secrets (never commit the key) or paste it in the sidebar. The scanner only fetches the latest candles.
- **CSV upload**: one file per instrument, e.g. a Kaggle XAUUSD / S&P 500 dataset or a broker/StrategyTune export. Works with or
  without a header, single datetime column or separate date + time columns (MT4/MT5), `dd.mm.yyyy` dates and unix timestamps.
  Candles of 15m or finer are converted to the selected 15m / 30m automatically; a 30m file cannot be used for 15m.
  Times are read as given (broker time is fine, there are no session rules in the logic).

## Backtest notes
- Entry at breakout candle close (or next candle open — sidebar option).
- If SL and TP fall inside one candle, SL is assumed first.
- Position size = risk % of current equity ÷ SL distance, so results are in R and equity compounds.
- Commission/slippage are zero by default; use the "Round-turn cost %" input to include them.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy on Streamlit Community Cloud: connect this repo, main file `app.py`.
