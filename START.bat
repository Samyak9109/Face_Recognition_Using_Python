@echo off
title Attendance System Launcher
echo.
echo  ==========================================
echo    Attendance System - One Click Start
echo  ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)

echo  Starting system...
echo  Frontend : http://localhost:3000
echo  Backend  : http://localhost:5000
echo.
echo  Close this window to stop everything.
echo.

python start.py

pause
