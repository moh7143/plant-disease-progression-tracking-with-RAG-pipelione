@echo off
title Plant Disease Detector - Backend with ngrok
color 0A

echo.
echo  =====================================================
echo    Plant Disease Detector Backend + ngrok Tunnel
echo  =====================================================
echo.

:: -------------------------------------------------------
:: CONFIGURATION - Edit these two values
:: -------------------------------------------------------
set NGROK_AUTHTOKEN=PASTE_YOUR_AUTH_TOKEN_HERE
set NGROK_DOMAIN=PASTE_YOUR_FREE_STATIC_DOMAIN_HERE
:: Example: set NGROK_DOMAIN=lucky-tiger-7890.ngrok-free.app
:: -------------------------------------------------------

:: Check if ngrok.exe exists
if not exist "ngrok.exe" (
    echo [ERROR] ngrok.exe not found in this folder!
    echo         Download from: https://ngrok.com/download
    echo         Place ngrok.exe in: %~dp0
    pause
    exit /b 1
)

:: Setup ngrok authtoken (only needed once)
echo [1/4] Setting up ngrok auth token...
ngrok.exe config add-authtoken %NGROK_AUTHTOKEN% >nul 2>&1
echo       Done.
echo.

:: Kill any existing ngrok/flask on these ports
echo [2/4] Cleaning up old processes...
taskkill /F /IM ngrok.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul
echo       Done.
echo.

:: Start Flask backend in a new window
echo [3/4] Starting Flask backend server...
start "Flask Backend" cmd /k "python app.py"
timeout /t 8 >nul
echo       Flask is running on http://127.0.0.1:5001
echo.

:: Start ngrok tunnel with static domain
echo [4/4] Starting ngrok tunnel...
if "%NGROK_DOMAIN%"=="PASTE_YOUR_FREE_STATIC_DOMAIN_HERE" (
    echo       [WARNING] No static domain set. Using random URL (changes each session).
    echo       Set NGROK_DOMAIN in this script to get a permanent URL!
    echo.
    start "ngrok Tunnel" cmd /k "ngrok.exe http 5001"
) else (
    start "ngrok Tunnel" cmd /k "ngrok.exe http --domain=%NGROK_DOMAIN% 5001"
    echo.
    echo  =====================================================
    echo   Your permanent backend URL:
    echo   https://%NGROK_DOMAIN%
    echo  =====================================================
    echo.
    echo  Set this in constants.dart:
    echo  static const String baseUrl = 'https://%NGROK_DOMAIN%';
    echo.
)

echo  Keep this window open while testing the app.
echo  Press any key to stop all services...
pause >nul

:: Cleanup on exit
taskkill /F /IM ngrok.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
echo Servers stopped.
