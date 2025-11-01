from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

from src.providers.mt5_provider import Mt5Provider, Mt5Config, Mt5Unavailable
from src.backtesting.result_schema import BacktestRow, append_row_to_csv, now_ts


class SMACross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        price = self.data.Close
        self.sma1 = self.I(SMA, price, self.n1)
        self.sma2 = self.I(SMA, price, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


_STRATEGIES = {
    "sma_cross": SMACross,
}


def _ensure_bt_df(df: pd.DataFrame) -> pd.DataFrame:
    req = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in OHLCV DataFrame")
    out = df.copy()
    out = out.sort_values('Datetime')
    out.set_index('Datetime', inplace=True)
    return out


def expectancy_pct_from_trades(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty:
        return 0.0
    wins = trades[trades['PnL'] > 0]
    losses = trades[trades['PnL'] <= 0]
    p_win = len(wins) / len(trades) if len(trades) else 0
    avg_win = wins['ReturnPct'].mean() if len(wins) else 0
    avg_loss = losses['ReturnPct'].mean() if len(losses) else 0
    return float((p_win * avg_win) + ((1 - p_win) * avg_loss))


def run_mt5_backtest(
    symbol: str,
    timeframe: str,
    start_ts: int,
    end_ts: int,
    strategy_name: str,
    params: Optional[Dict[str, Any]],
    stats_csv: Path,
    out_base: Path,
) -> Dict[str, Any]:
    if strategy_name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    cfg = Mt5Config.from_env()
    provider = Mt5Provider(cfg)
    provider.connect()

    ohlcv = provider.get_ohlcv(symbol, timeframe, start_ts, end_ts)
    bt_df = _ensure_bt_df(ohlcv)

    StrategyCls = _STRATEGIES[strategy_name]
    if params:
        for k, v in params.items():
            if hasattr(StrategyCls, k):
                setattr(StrategyCls, k, v)

    bt = Backtest(bt_df, StrategyCls, cash=100_000, commission=0.0005)
    stats = bt.run()

    # Extract stats
    ret = float(stats.get('Return [%]', 0.0))
    bh = float(stats.get('Buy & Hold Return [%]', 0.0))
    mdd = float(stats.get('Max. Drawdown [%]', 0.0))
    sharpe = float(stats.get('Sharpe Ratio', 0.0))
    sortino = float(stats.get('Sortino Ratio', 0.0))
    exposure = float(stats.get('Exposure [%]', 0.0))
    trades_count = int(stats.get('# Trades', 0))

    # Trades with returns
    trades = None
    try:
        trades = stats._trades  # Backtesting.py exposes trades on result Series
    except Exception:
        trades = getattr(stats, 'trades', None)
    trades = trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    if not trades.empty:
        # Backtesting library column names may vary; ensure ReturnPct exists
        if 'ReturnPct' not in trades.columns:
            # approximate from 'Return [%]' if present
            if 'Return [%]' in trades.columns:
                trades['ReturnPct'] = trades['Return [%]']
            elif 'PnL' in trades.columns and 'EntryPrice' in trades.columns:
                trades['ReturnPct'] = (trades['PnL'] / trades['EntryPrice']) * 100.0
            else:
                trades['ReturnPct'] = 0.0
    exp_pct = expectancy_pct_from_trades(trades)

    ts = now_ts()
    run_dir = out_base / 'backtests' / 'mt5' / f"{strategy_name}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    trades_path = run_dir / 'trades.csv'
    eq_path = run_dir / 'equity.csv'
    stats_path = run_dir / 'stats.json'
    chart_path = run_dir / 'chart.html'

    try:
        trades.to_csv(trades_path, index=False)
    except Exception:
        pass

    try:
        eq = stats['_equity_curve'] if '_equity_curve' in stats else None
        if eq is not None:
            eq.to_csv(eq_path, index=False)
    except Exception:
        pass

    try:
        # Saves interactive HTML
        bt.plot(filename=str(chart_path), open_browser=False)
    except Exception:
        pass

    try:
        import json
        with open(stats_path, 'w') as f:
            json.dump({k: (float(v) if hasattr(v, '__float__') else v) for k, v in stats.items()}, f, default=str)
    except Exception:
        pass

    # Append to CSV for dashboard table
    row = BacktestRow(
        strategy_name=strategy_name,
        thread_id='mt5',
        return_pct=ret,
        buyhold_pct=bh,
        max_dd_pct=mdd,
        sharpe=sharpe,
        sortino=sortino,
        exposure_pct=exposure,
        expectancy_pct=exp_pct,
        trades=trades_count,
        file_path=str(chart_path),
        data=f"mt5:{timeframe}:{symbol}",
        time=ts,
    )
    append_row_to_csv(stats_csv, row)

    return {
        "run_dir": str(run_dir),
        "stats": {
            "return_pct": ret,
            "buyhold_pct": bh,
            "max_dd_pct": mdd,
            "sharpe": sharpe,
            "sortino": sortino,
            "exposure_pct": exposure,
            "expectancy_pct": exp_pct,
            "trades": trades_count,
        },
        "artifacts": {
            "trades_csv": str(trades_path),
            "equity_csv": str(eq_path),
            "chart_html": str(chart_path),
            "stats_json": str(stats_path),
        }
    }
