@echo off
title Threatora Cyber Console Launcher
color 0B
cls
echo ======================================================================
echo             THREATORA // TACTICAL CYBER CLI CLIENT
echo                   NTRO Problem Statement 26153
echo ======================================================================
echo.

set TARGET_URL=%1

if "%TARGET_URL%"=="" (
    echo [*] Connect this CLI to Threatora Server on another laptop or cloud.
    echo     Examples:
    echo       - Local PC:     127.0.0.1
    echo       - Other Laptop: 172.16.190.142
    echo       - Cloud/Tunnel: https://your-server.onrender.com
    echo.
    set /p INPUT_HOST="Enter Server IP or URL [Default: http://127.0.0.1:5000]: "
) else (
    set INPUT_HOST=%TARGET_URL%
)

if "%INPUT_HOST%"=="" set INPUT_HOST=http://127.0.0.1:5000

:: Ensure proper prefix and port if bare IP or host was provided
echo %INPUT_HOST% | findstr /i "^http://" >nul
if errorlevel 1 (
    echo %INPUT_HOST% | findstr /i "^https://" >nul
    if errorlevel 1 (
        echo %INPUT_HOST% | findstr ":" >nul
        if errorlevel 1 (
            set FINAL_URL=http://%INPUT_HOST%:5000
        ) else (
            set FINAL_URL=http://%INPUT_HOST%
        )
    ) else (
        set FINAL_URL=%INPUT_HOST%
    )
) else (
    set FINAL_URL=%INPUT_HOST%
)

echo.
echo [+] Target Threatora API: %FINAL_URL%
echo [*] Default Operator Credentials: admin / Threatora@2026
echo [*] Launching Interactive Tactical Console...
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
python -c "import cmd2" >nul 2>&1
if %errorlevel% equ 0 (
    set PY_BIN=python
    goto :python_ready
)

:: Test py launcher variants
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 -c "import cmd2" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py -3.11
        goto :python_ready
    )
    py -3.13 -c "import cmd2" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py -3.13
        goto :python_ready
    )
    py -c "import cmd2" >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_BIN=py
        goto :python_ready
    )
)

set PY_BIN=python

:python_ready
%PY_BIN% cli.py console --api-url %FINAL_URL%

pause

