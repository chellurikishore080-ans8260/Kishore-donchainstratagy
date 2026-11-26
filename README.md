# Donchian Breakout Strategy — Scanner + Backtest

Streamlit app for **Gold, Silver, US500** on **5m / 15m / 30m / 1h / 4h**.

## Rules
**LONG**
1. Donchian upper of the previous N candles (default 20, entry candle excluded)
2. Bullish candle closes above that upper band
3. EMA fast > EMA slow
4. EMA slow rising

**SHORT** — mirror image (bearish candle closes below Donchian lower, EMA fast < EMA slow, EMA slow falling).

**SL**: breakout candle low (long) / high (short)
**Risk**: 0.5% per trade, skip the trade if SL distance > 1% of entry, one position at a time
**TP experiments**: 2R, 3R, 50% @ 1R + 50% @ 2R, 50% @ 1R + 50% @ 3R

## EMA presets (for lower timeframes)
| Preset | Fast | Slow |
|---|---|---|
| Original | 50 | 100 |
| Medium | 30 | 60 |
| Fast | 20 | 50 |
| Fastest | 10 | 30 |

Use the sidebar to pick a preset or enter custom lengths. The **EMA / TP sweep** tab tests every
preset × Donchian length × TP mode on all selected instruments and timeframes in one go.

## Tabs
- **Scanner** — latest signal on each instrument/timeframe (entry, SL, SL %, 1R/2R/3R targets, VALID / SKIP).
- **Backtest** — all four TP experiments side by side, equity curve, chart with entries, trades table + CSV.
- **EMA / TP sweep** — ranked comparison table.

## Data sources
- **Yahoo Finance** (default): Gold `GC=F`, Silver `SI=F`, US500 `ES=F`. Only ~60 days of 5m/15m history.
- **Twelve Data**: longer intraday history (API key; add `TWELVE_DATA_KEY` in Streamlit secrets or paste in the sidebar).
- **CSV upload**: export from your broker/StrategyTune (time, open, high, low, close) for the longest history.

## Backtest notes
- Entry at breakout candle close (or next candle open — sidebar option).
- If SL and TP fall inside one candle, SL is assumed first.
- Position size = risk % of current equity ÷ SL distance, so results are in R and equity compounds.
- Commission/slippage are zero by default; use the "Round-turn cost %" input to include them.
- Optional: move SL to breakeven after the 1R partial.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy on Streamlit Community Cloud: connect this repo, main file `app.py`.
