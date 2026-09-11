"""Unit and Integration Tests for Threatora Zero-Trust Authentication."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.app import app
from src.db.session import init_db


def test_authentication_workflow():
    print("=" * 70)
    print(" [+] THREATORA ZERO-TRUST AUTHENTICATION TEST SUITE")
    print("=" * 70)

    init_db()
    client = app.test_client()

    # 1. Unauthenticated request to / should redirect to /login
    print("[*] 1. Testing Unauthenticated Route Protection...")
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 302, f"Expected 302 redirect, got {res.status_code}"
    assert "/login" in res.headers["Location"], f"Expected /login redirect, got {res.headers['Location']}"
    print("  [+] Unauthenticated request successfully redirected to /login.")

    # 2. GET /login should render the login page
    print("[*] 2. Testing Login Page Rendering...")
    res = client.get("/login")
    assert res.status_code == 200
    assert b"Sign In to Operations Center" in res.data
    assert b"Threatora" in res.data
    print("  [+] GET /login rendered successfully with clean SaaS portal.")

    # 3. Invalid credentials attempt
    print("[*] 3. Testing Invalid Credentials Rejection...")
    res = client.post("/login", data={"username": "admin", "password": "WrongPassword!"})
    assert res.status_code == 401 or res.status_code == 200
    assert b"Invalid Operator Credentials" in res.data
    print("  [+] Invalid credentials correctly rejected.")

    # 4. Successful Login
    print("[*] 4. Testing Valid Operator Login (admin / Threatora@2026)...")
    res = client.post(
        "/login",
        data={"username": "admin", "password": "Threatora@2026"},
        follow_redirects=True
    )
    assert res.status_code == 200
    assert b"Alex Kelly" in res.data
    assert b"Threatora // Network Attack Forecasting" in res.data
    print("  [+] Operator successfully authenticated and session initialized.")

    # 5. Verify /api/v1/auth/me with session
    print("[*] 5. Testing /api/v1/auth/me Profile Inspection...")
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 200
    data = res.get_json()
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "CHIEF_CISO_ADMIN"
    print(f"  [+] Session verified for operator: {data['user']['full_name']} ({data['user']['role']})")

    # 6. Testing System API Key (Headless CLI / Tests)
    print("[*] 6. Testing Headless System API Key Access...")
    unauth_client = app.test_client()
    res = unauth_client.get("/api/v1/auth/me", headers={"X-API-Key": "threatora-zero-trust"})
    assert res.status_code == 200
    print("  [+] System API Key authorized successfully.")

    # 7. Testing Logout
    print("[*] 7. Testing Operator Session Termination (/logout)...")
    res = client.get("/logout", follow_redirects=False)
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    # Verify session is cleared
    res_after = client.get("/", follow_redirects=False)
    assert res_after.status_code == 302
    print("  [+] Session cleanly terminated and user redirected back to login.")

    print("\n" + "=" * 70)
    print(" [+] ALL AUTHENTICATION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    test_authentication_workflow()
