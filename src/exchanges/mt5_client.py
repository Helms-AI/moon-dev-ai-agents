"""
🌙 Moon Dev's MT5 HTTP Client
Add-only MT5 bridge client for trading metals/FX via REST.
"""

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, Optional
from datetime import datetime

import requests
from requests import Response
from termcolor import cprint
from dotenv import load_dotenv


class MT5Client:
    """Thin wrapper around the MT5 bridge OpenAPI.

    Environment variables used:
      - MT5_API_BASE
      - MT5_API_USER, MT5_API_PASSWORD (API auth for /auth/login)
      - MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (MT5 terminal account)
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 mt5_login: Optional[str] = None,
                 mt5_password: Optional[str] = None,
                 mt5_server: Optional[str] = None):
        # Load .env from project root explicitly (src/exchanges/../../.env)
        try:
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            load_dotenv(dotenv_path=project_root / '.env')
        except Exception:
            load_dotenv()

        self.base_url = (base_url or os.getenv("MT5_API_BASE_URL") or os.getenv("MT5_API_BASE") or "https://mt5.atm.helms.ai").rstrip("/")
        # Support common alias env var names
        self.username = (username or os.getenv("MT5_API_USER") or os.getenv("MT5_API_USERNAME") or os.getenv("MT5_USERNAME"))
        self.password = (password or os.getenv("MT5_API_PASSWORD") or os.getenv("MT5_API_PASS") or os.getenv("MT5_API_PWD") or os.getenv("MT5_PASSWORD"))
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Login to API
        if not self.username or not self.password:
            raise ValueError("MT5 API credentials not provided (MT5_API_USER/MT5_API_PASSWORD)")
        self._login_api()

        # Optional: login to MT5 terminal session (uses env defaults if omitted)
        self._mt5_login = mt5_login or os.getenv("MT5_LOGIN") or os.getenv("MT5_LOGIN_ID") or os.getenv("MT5_ACCOUNT")
        self._mt5_password = mt5_password or os.getenv("MT5_PASSWORD") or os.getenv("MT5_PASS")
        self._mt5_server = mt5_server or os.getenv("MT5_SERVER") or os.getenv("MT5_BROKER_SERVER")
        try:
            self.login_mt5(self._mt5_login, self._mt5_password, self._mt5_server)
        except Exception as e:
            # Do not hard-fail if session already initialized server-side
            cprint(f"⚠️ MT5 session login note: {e}", "yellow")

    # ------------------------- Internal helpers -------------------------
    def _login_api(self) -> None:
        url = f"{self.base_url}/auth/login"
        payload = {"username": self.username, "password": self.password}
        resp = self.session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data.get("access_token")
        self._refresh_token = data.get("refresh_token")
        if not self._access_token:
            raise RuntimeError("No access_token returned from MT5 API login")
        self.session.headers.update({"Authorization": f"Bearer {self._access_token}"})
        cprint("✅ MT5 API auth succeeded", "green")

    def _refresh_api_token(self) -> None:
        if not self._refresh_token:
            self._login_api()
            return
        url = f"{self.base_url}/auth/refresh"
        payload = {"refresh_token": self._refresh_token}
        resp = self.session.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            self._access_token = data.get("access_token")
            self._refresh_token = data.get("refresh_token")
            self.session.headers.update({"Authorization": f"Bearer {self._access_token}"})
            cprint("🔁 MT5 API token refreshed", "cyan")
        else:
            cprint("⚠️ MT5 API token refresh failed; re-logging in", "yellow")
            self._login_api()

    def _request(self, method: str, path: str, *, params: Dict[str, Any] | None = None,
                 json: Dict[str, Any] | None = None, timeout: int = 60) -> Response:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
        if resp.status_code == 401:
            with self._token_lock:
                # Double-check and refresh
                self._refresh_api_token()
            resp = self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
        resp.raise_for_status()
        return resp

    # ------------------------- Public MT5 endpoints -------------------------
    def login_mt5(self, login: Optional[str], password: Optional[str], server: Optional[str]) -> Dict[str, Any]:
        payload = {
            "login": int(login) if (login and str(login).isdigit()) else None,
            "password": password,
            "server": server,
            "timeout": 60000,
        }
        resp = self._request("POST", "/mt5/login", json=payload)
        return resp.json()

    def account_info(self) -> Dict[str, Any]:
        return self._request("GET", "/mt5/account_info").json()

    def symbol_select(self, symbol: str, enable: bool = True) -> Dict[str, Any]:
        payload = {"symbol": symbol, "enable": enable}
        return self._request("POST", "/mt5/symbol_select", json=payload).json()

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/mt5/symbol_info", params={"symbol": symbol}).json()

    def symbol_tick(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/mt5/symbol_info_tick", params={"symbol": symbol}).json()

    def copy_rates_range(self, symbol: str, timeframe: str, date_from: str, date_to: str) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "date_from": date_from,
            "date_to": date_to,
        }
        return self._request("POST", "/mt5/copy_rates_range", json=payload).json()

    def positions_get(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/mt5/positions_get", params=params).json()

    def orders_get(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/mt5/orders_get", params=params).json()

    def order_check(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/mt5/order_check", json=request).json()

    def order_send(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/mt5/order_send", json=request).json()