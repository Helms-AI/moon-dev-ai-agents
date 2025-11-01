from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from datetime import datetime

CSV_COLUMNS = [
    'Strategy Name', 'Thread ID', 'Return %', 'Buy & Hold %',
    'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio', 'Exposure %',
    'EV %', 'Trades', 'File Path', 'Data', 'Time'
]

@dataclass
class BacktestRow:
    strategy_name: str
    thread_id: str
    return_pct: float
    buyhold_pct: float
    max_dd_pct: float
    sharpe: float
    sortino: float
    exposure_pct: float
    expectancy_pct: float
    trades: int
    file_path: str
    data: str
    time: str

    def to_series(self) -> pd.Series:
        return pd.Series({
            'Strategy Name': self.strategy_name,
            'Thread ID': self.thread_id,
            'Return %': self.return_pct,
            'Buy & Hold %': self.buyhold_pct,
            'Max Drawdown %': self.max_dd_pct,
            'Sharpe Ratio': self.sharpe,
            'Sortino Ratio': self.sortino,
            'Exposure %': self.exposure_pct,
            'EV %': self.expectancy_pct,
            'Trades': self.trades,
            'File Path': self.file_path,
            'Data': self.data,
            'Time': self.time,
        })


def append_row_to_csv(csv_path: Path, row: BacktestRow) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = row.to_series().to_frame().T

    if csv_path.exists():
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Write header
        df_row.to_csv(csv_path, mode='w', header=True, index=False)


def now_ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
