from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from src.providers.mt5_provider import Mt5Provider, Mt5Config, Mt5Unavailable
from src.backtesting.mt5_backtest import run_mt5_backtest

router = APIRouter()

SRC_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_BASE_DIR = SRC_DIR / "data" / "rbi_pp_multi"
STATS_CSV = TEMPLATE_BASE_DIR / "backtest_stats.csv"
OUT_BASE = TEMPLATE_BASE_DIR

running_jobs: Dict[str, Dict[str, Any]] = {}

def _json_safe(obj):
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

class MT5RunRequest(BaseModel):
    symbol: str
    timeframe: str = Field(pattern="^(M1|M5|M15|M30|H1|H4|D1)$")
    start: int  # epoch seconds
    end: int    # epoch seconds
    strategy: str = "sma_cross"
    params: Optional[Dict[str, Any]] = None
    run_name: Optional[str] = None


@router.get("/api/mt5/timeframes")
async def mt5_timeframes():
    prov = Mt5Provider(Mt5Config.from_env())
    return {"timeframes": prov.list_timeframes()}


@router.get("/api/mt5/symbols")
async def mt5_symbols(q: Optional[str] = None):
    try:
        prov = Mt5Provider(Mt5Config.from_env())
        prov.connect()
        return {"available": True, "symbols": prov.list_symbols(q)}
    except Mt5Unavailable as e:
        return {"available": False, "symbols": [], "error": str(e)}
    except Exception as e:
        # Ensure UI gets a JSON error instead of a 500
        return {"available": False, "symbols": [], "error": str(e)}


@router.post("/api/backtest/mt5/run")
async def mt5_run(req: MT5RunRequest, background: BackgroundTasks):
    run_name = req.run_name or f"mt5_{req.strategy}_{req.symbol}_{req.timeframe}_{int(datetime.now().timestamp())}"
    running_jobs[run_name] = {"status": "queued"}

    def job():
        try:
            running_jobs[run_name] = {"status": "running"}
            result = run_mt5_backtest(
                symbol=req.symbol,
                timeframe=req.timeframe,
                start_ts=req.start,
                end_ts=req.end,
                strategy_name=req.strategy,
                params=req.params,
                stats_csv=STATS_CSV,
                out_base=OUT_BASE,
            )
            running_jobs[run_name] = {"status": "complete", "result": _json_safe(result)}
        except Exception as e:
            running_jobs[run_name] = {"status": "error", "error": str(e)}

    background.add_task(job)
    return {"run_name": run_name, "status": "started"}


@router.get("/api/backtest/mt5/status/{run_name}")
async def mt5_status(run_name: str):
    return running_jobs.get(run_name, {"status": "not_found"})
