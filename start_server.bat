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
:: Detect Python executable with dependencies installed
set PY_BIN=
if exist "venv\Scripts\python.exe" (
    set PY_BIN=venv\Scripts\python.exe
    goto :python_ready
)
if exist ".venv\Scripts\python.exe" (
    set PY_BIN=.venv\Scripts\python.exe
    goto :python_ready
)

:: Test standard python
python -c "import flask" >nul 2>&1
if %errorlevel% equ 0 (
    set PY_BIN=python
    goto :python_ready
)

:: Test py launcher variants
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 -c "import flask" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py -3.11
        goto :python_ready
    )
    py -3.13 -c "import flask" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py -3.13
        goto :python_ready
    )
    py -c "import flask" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py
        goto :python_ready
    )
)

set PY_BIN=python

:python_ready
%PY_BIN% app.py
pause

