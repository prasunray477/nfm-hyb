import socket
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.data_service import DataFetchError, DataService


HOSTS = (
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "fc.yahoo.com",
)


def check_tcp(host: str, port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=10):
            return True
    except OSError as exc:
        print(f"[FAIL] TCP {host}:{port} -> {exc}")
        return False


def main() -> int:
    reachable = [check_tcp(host) for host in HOSTS]
    if not any(reachable):
        print(
            "\nPython cannot reach Yahoo Finance. Allow python.exe through "
            "your firewall/VPN/proxy, or set YFINANCE_PROXY in .env."
        )
        return 1

    try:
        prices = DataService.fetch("AAPL", "5y")
    except DataFetchError as exc:
        print(f"\nYahoo connection failed: {exc}")
        return 1

    print(f"\nMarket data OK: fetched {len(prices)} AAPL prices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
