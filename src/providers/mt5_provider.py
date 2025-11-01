from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover
    mt5 = None


class Mt5Unavailable(Exception):
    pass


_TIMEFRAMES = {
    "M1": getattr(mt5, "TIMEFRAME_M1", 1) if mt5 else "M1",
    "M5": getattr(mt5, "TIMEFRAME_M5", 5) if mt5 else "M5",
    "M15": getattr(mt5, "TIMEFRAME_M15", 15) if mt5 else "M15",
    "M30": getattr(mt5, "TIMEFRAME_M30", 30) if mt5 else "M30",
    "H1": getattr(mt5, "TIMEFRAME_H1", 60) if mt5 else "H1",
    "H4": getattr(mt5, "TIMEFRAME_H4", 240) if mt5 else "H4",
    "D1": getattr(mt5, "TIMEFRAME_D1", 1440) if mt5 else "D1",
}


@dataclass
class Mt5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    http_base: Optional[str] = None  # If using an external bridge
    http_user: Optional[str] = None
    http_password: Optional[str] = None

    @staticmethod
    def from_env() -> "Mt5Config":
        login = os.getenv("MT5_LOGIN")
        # Support both MT5_HTTP_* and MT5_API_* naming conventions
        http_base = os.getenv("MT5_HTTP_BASE") or os.getenv("MT5_API_BASE_URL")
        http_user = os.getenv("MT5_HTTP_USER") or os.getenv("MT5_API_USERNAME")
        http_pass = os.getenv("MT5_HTTP_PASSWORD") or os.getenv("MT5_API_PASSWORD")
        return Mt5Config(
            login=int(login) if login else None,
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            http_base=http_base,
            http_user=http_user,
            http_password=http_pass,
        )


class Mt5Provider:
    def __init__(self, cfg: Mt5Config):
        self.cfg = cfg
        self._mode = None  # "native" or "http"
        self._headers = {}
        self._symbols_path = "/mt5/symbols_get"
        self._ohlcv_path = "/mt5/copy_rates_range"
        self._ohlcv_from_path = "/mt5/copy_rates_from"
        self._login_path = "/auth/login"

    @staticmethod
    def available() -> bool:
        return bool(mt5) or bool(os.getenv("MT5_HTTP_BASE"))

    def connect(self) -> None:
        # Prefer native if possible
        if mt5 is not None:
            if not mt5.initialize():
                raise Mt5Unavailable(f"MT5 initialize failed: {mt5.last_error()}")
            if self.cfg.login and self.cfg.password and self.cfg.server:
                if not mt5.login(self.cfg.login, password=self.cfg.password, server=self.cfg.server):
                    raise Mt5Unavailable(f"MT5 login failed: {mt5.last_error()}")
            self._mode = "native"
            return

        # Fallback to HTTP bridge
        if self.cfg.http_base:
            self._mode = "http"
            # Perform JWT login if credentials provided
            if self.cfg.http_user and self.cfg.http_password:
                import requests
                url = self._join(self.cfg.http_base, self._login_path)
                r = requests.post(url, json={"username": self.cfg.http_user, "password": self.cfg.http_password}, timeout=20)
                r.raise_for_status()
                data = r.json()
                token = data.get("access_token")
                if not token:
                    raise Mt5Unavailable("Login succeeded but no access_token returned")
                self._headers = {"Authorization": f"Bearer {token}"}
            return

        raise Mt5Unavailable("No MT5 backend available (native module or MT5_HTTP_BASE)")

    def list_timeframes(self) -> List[str]:
        return list(_TIMEFRAMES.keys())

    def _join(self, base: str, path: str) -> str:
        if not base:
            return path
        if base.endswith('/'):
            base = base[:-1]
        if not path.startswith('/'):
            path = '/' + path
        return base + path

    def list_symbols(self, search: Optional[str] = None) -> List[str]:
        if self._mode == "native":
            symbols = [s.name for s in mt5.symbols_get()]
            if search:
                symbols = [s for s in symbols if search.upper() in s.upper()]
            return symbols
        elif self._mode == "http":
            import requests
            url = self._join(self.cfg.http_base, self._symbols_path)
            params = {}
            if search:
                # Map to group filter when available; pattern match by wildcard
                params["group"] = f"*{search}*"
            r = requests.get(url, timeout=30, headers=self._headers or None, params=params)
            r.raise_for_status()
            payload = r.json()
            # Accept either list of strings or list of dicts
            if isinstance(payload, list):
                if payload and isinstance(payload[0], dict):
                    symbols = [d.get('name') or d.get('Symbol') or d.get('symbol') for d in payload]
                    symbols = [s for s in symbols if s]
                else:
                    symbols = [str(x) for x in payload]
            elif isinstance(payload, dict) and 'symbols' in payload:
                symbols = payload['symbols']
            else:
                symbols = []
            if search:
                symbols = [s for s in symbols if search.upper() in s.upper()]
            return symbols
        else:
            raise Mt5Unavailable("Provider not connected")

    def get_ohlcv(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        if timeframe not in _TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if self._mode == "native":
            tf = _TIMEFRAMES[timeframe]
            rates = mt5.copy_rates_range(symbol, tf, start_ts, end_ts)
            if rates is None:
                err = mt5.last_error()
                raise Mt5Unavailable(f"copy_rates_range failed: {err}")
            df = pd.DataFrame(rates)
            # MT5 returns 'time' in seconds since epoch
            df['Datetime'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

        elif self._mode == "http":
            import requests
            from datetime import datetime, timezone
            def iso(ts):
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace('+00:00','Z')

            # Prefer copy_rates_from for robust behavior
            seconds_per = {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
            cnt = max(1, int((int(end_ts) - int(start_ts)) / seconds_per.get(timeframe, 60)) + 1)
            url = self._join(self.cfg.http_base, self._ohlcv_from_path)
            body = {"symbol": symbol, "timeframe": timeframe, "date_from": iso(start_ts), "count": cnt}
            r = requests.post(url, json=body, timeout=90, headers=self._headers or None)

            # If endpoint not found or server error, try range endpoint as fallback
            if r.status_code >= 400:
                url2 = self._join(self.cfg.http_base, self._ohlcv_path)
                body2 = {"symbol": symbol, "timeframe": timeframe, "date_from": iso(start_ts), "date_to": iso(end_ts)}
                r = requests.post(url2, json=body2, timeout=90, headers=self._headers or None)

            r.raise_for_status()
            payload = r.json()
            data = payload.get('data') if isinstance(payload, dict) else payload
            df = pd.DataFrame(data)
            if df.empty:
                raise Mt5Unavailable("No OHLCV returned from bridge")
            # Normalize columns
            if 'Datetime' not in df.columns:
                if 'time' in df.columns:
                    # time may be seconds; if ms, adjust automatically
                    df['Datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
                    if df['Datetime'].isna().all():
                        df['Datetime'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                elif 'date' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            # value columns
            rename = {}
            for src, dst in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('tick_volume','Volume'),('real_volume','Volume'),('volume','Volume')]:
                if src in df.columns and dst not in df.columns:
                    rename[src]=dst
            df.rename(columns=rename, inplace=True)
            cols = ['Datetime','Open','High','Low','Close','Volume']
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise Mt5Unavailable(f"Bridge OHLCV missing columns: {missing}")
            return df[cols]

        else:
            raise Mt5Unavailable("Provider not connected")
