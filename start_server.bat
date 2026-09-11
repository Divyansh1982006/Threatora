@echo off
title Threatora Central Core Server
color 0A
cls
echo ======================================================================
echo           THREATORA // CENTRAL ZERO-TRUST DEFENSE SERVER
echo                   NTRO Problem Statement 26153
echo ======================================================================
echo.
echo [*] Detecting Local Network IP for Multi-Laptop Access...
for /f "tokens=4" %%a in ('route print ^| find " 0.0.0.0 "') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
echo [+] Detected Network Gateway Interface: %LOCAL_IP%
echo.
echo [!] ACCESS URLS FOR OTHER LAPTOPS / CLIENTS ON SAME WI-FI / LAN:
echo     ------------------------------------------------------------------
echo     Web Portal (Browser)   : http://%LOCAL_IP%:5000
echo     CLI Remote Command     : python cli.py console --api-url http://%LOCAL_IP%:5000
echo     Local Host URL         : http://127.0.0.1:5000
echo     Default Operator Login : admin / Threatora@2026
echo     ------------------------------------------------------------------
echo.
echo [*] Launching Threatora WSGI Server (Binding 0.0.0.0:5000)...
echo [*] Press CTRL+C to stop the server.
echo.
python server/app.py
pause
