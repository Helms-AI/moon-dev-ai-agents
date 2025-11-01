"""
🌙 Moon Dev's MT5 Exchange Adapter (add-only)
Provides TradingAgent-compatible functions for MT5 symbols (e.g., XAUUSD).
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

from src.exchanges.mt5_client import MT5Client

load_dotenv()

# Initialize client (env-driven)
_client = MT5Client()

# Simple in-memory symbol meta cache
_symbol_meta: Dict[str, Dict[str, float]] = {}


def _norm_get(d: Dict[str, Any], *names: str, default=None):
    if not isinstance(d, dict):
        return default
    lower_map = {str(k).lower(): k for k in d.keys()}
    for name in names:
        key = name.lower()
        if key in lower_map:
            return d[lower_map[key]]
    # Try stripping underscores/case-insensitive search
    for k in d.keys():
        if str(k).replace("_", "").lower() in {n.replace("_", "").lower() for n in names}:
            return d[k]
    return default


def _map_timeframe(tf: str) -> str:
    tf = str(tf).upper()
    mapping = {
        "1M": "M1", "3M": "M3", "5M": "M5", "15M": "M15", "30M": "M30",
        "1H": "H1", "2H": "H2", "4H": "H4", "6H": "H6", "8H": "H8", "12H": "H12",
        "1D": "D1", "3D": "D1",  # use daily and resample externally if needed
        "1W": "W1", "1MO": "MN1", "1MTH": "MN1",
    }
    if tf in mapping:
        return mapping[tf]
    # Already MT5 format?
    allowed = {"M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
               "H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"}
    return tf if tf in allowed else "H1"


def _ensure_symbol(symbol: str) -> Dict[str, Any]:
    # Ensure symbol is in MarketWatch and cache meta
    _client.symbol_select(symbol, True)
    info = _client.symbol_info(symbol)
    if not info:
        raise RuntimeError(f"symbol_info failed for {symbol}")

    # Extract common fields (case-insensitive)
    contract_size = float(_norm_get(info, "trade_contract_size", "contract_size", default=100.0))
    volume_min = float(_norm_get(info, "volume_min", default=0.01))
    volume_max = float(_norm_get(info, "volume_max", default=1000.0))
    volume_step = float(_norm_get(info, "volume_step", default=0.01))
    digits = int(float(_norm_get(info, "digits", default=2)))
    point = float(_norm_get(info, "point", default=10 ** (-digits)))

    _symbol_meta[symbol] = {
        "contract_size": contract_size,
        "volume_min": volume_min,
        "volume_max": volume_max,
        "volume_step": volume_step,
        "digits": digits,
        "point": point,
    }
    return info


def _get_meta(symbol: str) -> Dict[str, float]:
    if symbol not in _symbol_meta:
        _ensure_symbol(symbol)
    return _symbol_meta[symbol]


def ask_bid(symbol: str) -> Tuple[float, float, Dict[str, Any]]:
    """Return (ask, bid, raw) for symbol."""
    tick = _client.symbol_tick(symbol)
    ask = float(_norm_get(tick, "ask", default=_norm_get(tick, "Ask", default=0)))
    bid = float(_norm_get(tick, "bid", default=_norm_get(tick, "Bid", default=0)))
    return ask, bid, tick


def token_price(symbol: str) -> float:
    ask, bid, _ = ask_bid(symbol)
    if ask and bid:
        return (ask + bid) / 2.0
    last = _norm_get(_client.symbol_tick(symbol), "last", "Last", default=0)
    return float(last) if last else 0.0


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def _calc_deviation_points(symbol: str, slippage_bps: int | float) -> int:
    # convert bps slippage into price deviation in points
    meta = _get_meta(symbol)
    point = meta["point"]
    price = token_price(symbol)
    if price <= 0 or point <= 0:
        return 10  # safe small default
    deviation_price = float(slippage_bps) / 10000.0 * price
    deviation_points = int(round(deviation_price / point))
    return max(5, deviation_points)


def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    meta = _get_meta(symbol)
    contract = meta["contract_size"]

    data = _client.positions_get(symbol=symbol)
    if not data:
        return None

    # Expect list/array of positions
    positions = data if isinstance(data, list) else data.get("data") or data.get("positions") or []
    if not positions:
        return None

    total_lots = 0.0
    weighted_entry_value = 0.0
    profit_total = 0.0

    for pos in positions:
        vol = float(_norm_get(pos, "volume", "volume_lots", default=0))
        typ = int(float(_norm_get(pos, "type", default=0)))  # 0=BUY, 1=SELL
        entry = float(_norm_get(pos, "price_open", default=0))
        profit = float(_norm_get(pos, "profit", default=0))
        sign = 1.0 if typ == 0 else -1.0
        lots_signed = sign * vol
        total_lots += lots_signed
        weighted_entry_value += abs(lots_signed) * entry
        profit_total += profit

    if abs(total_lots) < 1e-9:
        return None

    entry_price = weighted_entry_value / abs(total_lots) if abs(total_lots) > 0 else 0.0
    mark = token_price(symbol)
    if mark <= 0:
        # fallback to current from first position
        mark = float(_norm_get(positions[0], "price_current", default=entry_price))

    # Compute pnl if not provided
    if abs(profit_total) < 1e-9:
        profit_total = (mark - entry_price) * (1 if total_lots > 0 else -1) * abs(total_lots) * contract

    denom = (abs(total_lots) * contract * entry_price) if (contract > 0 and entry_price > 0) else 0.0
    pnl_pct = (profit_total / denom * 100.0) if denom > 0 else 0.0

    return {
        "has_position": True,
        "position_amount": total_lots,  # signed lots
        "entry_price": entry_price,
        "mark_price": mark,
        "pnl": profit_total,
        "pnl_percentage": pnl_pct,
        "is_long": total_lots > 0,
        "raw_data": positions,
    }


def get_token_balance_usd(symbol: str) -> float:
    meta = _get_meta(symbol)
    pos = get_position(symbol)
    if not pos:
        return 0.0
    lots = abs(float(pos.get("position_amount", 0)))
    price = float(pos.get("mark_price", token_price(symbol)))
    return lots * meta["contract_size"] * price


def get_account_balance() -> Dict[str, float]:
    """Return MT5 account balances (equity/balance/margin_free)."""
    try:
        info = _client.account_info()
        equity = float(_norm_get(info, "equity", default=_norm_get(info, "Equity", default=0)))
        balance = float(_norm_get(info, "balance", default=_norm_get(info, "Balance", default=0)))
        free = float(_norm_get(info, "margin_free", default=_norm_get(info, "MarginFree", default=0)))
        return {"equity": equity, "balance": balance, "margin_free": free}
    except Exception as e:
        cprint(f"❌ Error reading MT5 account_info: {e}", "red")
        return {"equity": 0.0, "balance": 0.0, "margin_free": 0.0}


def market_buy(symbol: str, usd_amount: float, slippage: int | float = 0, leverage: int | None = None) -> Dict[str, Any] | None:
    meta = _get_meta(symbol)
    price = token_price(symbol)
    if price <= 0:
        cprint(f"❌ {symbol} price unavailable", "red")
        return None

    lots_raw = float(usd_amount) / (meta["contract_size"] * price)
    lots = _round_to_step(lots_raw, meta["volume_step"])
    lots = max(lots, meta["volume_min"])
    lots = min(lots, meta["volume_max"])
    if lots <= 0:
        cprint(f"❌ Computed lot size is zero for {symbol}", "red")
        return None

    deviation = _calc_deviation_points(symbol, slippage or 0)
    req = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": lots,
        "type": 0,   # BUY (market)
        "price": price,
        "deviation": deviation,
        "type_filling": 1,  # IOC
        "type_time": 0,     # GTC
        "comment": "moon-dev-agent",
    }
    return _client.order_send(req)


def market_sell(symbol: str, usd_amount: float, slippage: int | float = 0, leverage: int | None = None) -> Dict[str, Any] | None:
    meta = _get_meta(symbol)
    price = token_price(symbol)
    if price <= 0:
        cprint(f"❌ {symbol} price unavailable", "red")
        return None

    lots_raw = float(usd_amount) / (meta["contract_size"] * price)
    lots = _round_to_step(lots_raw, meta["volume_step"])
    lots = max(lots, meta["volume_min"])
    lots = min(lots, meta["volume_max"])
    if lots <= 0:
        cprint(f"❌ Computed lot size is zero for {symbol}", "red")
        return None

    deviation = _calc_deviation_points(symbol, slippage or 0)
    req = {
        "action": 1,  # DEAL
        "symbol": symbol,
        "volume": lots,
        "type": 1,   # SELL (market)
        "price": price,
        "deviation": deviation,
        "type_filling": 1,  # IOC
        "type_time": 0,     # GTC
        "comment": "moon-dev-agent",
    }
    return _client.order_send(req)


def chunk_kill(symbol: str, max_usd_order_size: float, slippage: int | float = 0) -> bool:
    try:
        cprint(f"\n🔪 Closing {symbol} position in chunks...", "cyan")
        while True:
            pos = get_position(symbol)
            if not pos:
                cprint("✨ Position already flat", "green")
                return True

            lots_remaining = abs(float(pos.get("position_amount", 0)))
            if lots_remaining <= 1e-9:
                cprint("✨ Position flat", "green")
                return True

            meta = _get_meta(symbol)
            price = float(pos.get("mark_price", token_price(symbol)))
            # compute lot chunk by usd
            lots_chunk_raw = float(max_usd_order_size) / (meta["contract_size"] * price)
            lots_chunk = _round_to_step(max(lots_chunk_raw, meta["volume_min"]), meta["volume_step"])
            lots_chunk = min(lots_chunk, lots_remaining)
            usd_chunk = lots_chunk * meta["contract_size"] * price

            cprint(f"🔄 Chunk SELL {lots_chunk:.2f} lots (~${usd_chunk:,.2f})", "yellow")
            market_sell(symbol, usd_chunk, slippage)
            time.sleep(1.0)
    except Exception as e:
        cprint(f"❌ Error in chunk_kill: {e}", "red")
        return False


def ai_entry(symbol: str, amount: float, max_chunk_size: Optional[float] = None, leverage: Optional[int] = None, use_limit: bool = False) -> bool:
    """Simple MT5 entry helper to reach a USD notional target by buying in chunks.

    Args:
        symbol: MT5 symbol (e.g., 'XAUUSD')
        amount: Total USD notional to buy
        max_chunk_size: Optional chunk size in USD; if None uses single order
        leverage: unused for MT5 spot-style entry (kept for interface compatibility)
        use_limit: ignored (market orders used)
    Returns:
        bool indicating whether orders were submitted without local errors
    """
    try:
        if amount <= 0:
            cprint(f"⚠️ ai_entry ignored non-positive amount: {amount}", "yellow")
            return False

        if max_chunk_size is None or amount <= max_chunk_size:
            resp = market_buy(symbol, amount, slippage=0)
            return bool(resp)

        # Multiple chunks
        num_chunks = int(amount // max_chunk_size) + (1 if amount % max_chunk_size > 1e-9 else 0)
        per = amount / num_chunks
        cprint(f"🎯 MT5 ai_entry: ${amount:.2f} in {num_chunks} chunks of ~${per:.2f}", "cyan")
        remaining = amount
        for i in range(num_chunks):
            usd = min(per, remaining)
            cprint(f"🔄 Chunk {i+1}/{num_chunks}: BUY ${usd:.2f}", "cyan")
            resp = market_buy(symbol, usd, slippage=0)
            if not resp:
                cprint("⚠️ Chunk buy returned no response", "yellow")
            remaining -= usd
            time.sleep(0.3)
        return True
    except Exception as e:
        cprint(f"❌ Error in ai_entry: {e}", "red")
        return False


def get_data(symbol: str, days_back: float = 3, timeframe: str = "1H", add_indicators: bool = True) -> pd.DataFrame:
    try:
        _ensure_symbol(symbol)
        tf = _map_timeframe(timeframe)
        date_to = datetime.utcnow()
        date_from = date_to - timedelta(days=float(days_back))
        payload = _client.copy_rates_range(
            symbol=symbol,
            timeframe=tf,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
        )
        bars = payload if isinstance(payload, list) else payload.get("data") or []
        if not bars:
            cprint(f"⚠️ No bars returned for {symbol} {tf}", "yellow")
            return pd.DataFrame()

        # Normalize to DataFrame
        records = []
        for b in bars:
            t = _norm_get(b, "time", "datetime", "date")
            if t is None:
                continue
            # time may be epoch seconds or ISO string
            if isinstance(t, (int, float)):
                ts = datetime.utcfromtimestamp(float(t))
            else:
                try:
                    ts = datetime.fromisoformat(str(t))
                except Exception:
                    continue
            op = float(_norm_get(b, "open", default=0))
            hi = float(_norm_get(b, "high", default=0))
            lo = float(_norm_get(b, "low", default=0))
            cl = float(_norm_get(b, "close", default=0))
            vol = float(_norm_get(b, "real_volume", "tick_volume", "volume", default=0))
            records.append({"Datetime (UTC)": ts, "Open": op, "High": hi, "Low": lo, "Close": cl, "Volume": vol})

        df = pd.DataFrame(records).sort_values("Datetime (UTC)")
        return df
    except Exception as e:
        cprint(f"❌ Error fetching data for {symbol}: {e}", "red")
        return pd.DataFrame()