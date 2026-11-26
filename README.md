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
**Risk**: 0.5% per trade, skip the trade if SL distance > 1% of entry, one position at a time
**TP**: full exit at **1:1.5**, **1:2** or **1:3** (no partials)

## Tabs
- **Scanner** — latest signal on each instrument/timeframe (entry, SL, SL %, TP 1:1.5 / 1:2 / 1:3, VALID / SKIP).
- **Backtest** — summary of Gold/US500 × 15m/30m × 3 TP ratios, then detail with equity curve, chart and trades CSV.

## Data sources
- **Yahoo Finance** (default): Gold `GC=F`, US500 `ES=F`. Only ~60 days of 15m/30m history.
- **Twelve Data**: longer intraday history (API key; add `TWELVE_DATA_KEY` in Streamlit secrets or paste in the sidebar).
- **CSV upload**: export from your broker/StrategyTune (time, open, high, low, close) for the longest history.

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
