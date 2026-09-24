"""Integration test script for QVOD PCAP upload against live or local server."""

from pathlib import Path
import pytest
import requests

BASE_URL = "http://127.0.0.1:5000"
headers = {"X-API-Key": "threatora-zero-trust"}


def test_qvod_upload_optional():
    """Skips gracefully if live dev server or sample file is absent."""
    pcap_path = Path("data/uploads/botnet-capture-20110816-qvod.pcap")
    if not pcap_path.exists():
        pytest.skip(f"PCAP file {pcap_path} not found.")

    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=1)
        if r.status_code != 200:
            pytest.skip("Live server on port 5000 not responding.")
    except Exception:
        pytest.skip("Live server on port 5000 not reachable.")

    with open(pcap_path, "rb") as f:
        r_up = requests.post(
            f"{BASE_URL}/api/upload",
            headers=headers,
            files={"file": ("botnet-capture-20110816-qvod.pcap", f, "application/vnd.tcpdump.pcap")},
        )
    assert r_up.status_code in (200, 503)


if __name__ == "__main__":
    pcap_path = Path("data/uploads/botnet-capture-20110816-qvod.pcap")
    if pcap_path.exists():
        with open(pcap_path, "rb") as f:
            r_up = requests.post(
                f"{BASE_URL}/api/upload",
                headers=headers,
                files={"file": ("botnet-capture-20110816-qvod.pcap", f, "application/vnd.tcpdump.pcap")},
            )
        print("Upload status:", r_up.status_code)
        data = r_up.json()
        print("Status:", data.get("status"))
