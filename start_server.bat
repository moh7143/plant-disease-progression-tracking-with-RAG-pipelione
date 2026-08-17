@echo off
echo ============================================
echo   Plant Disease Detector - Backend Starter
echo ============================================
echo.

:: Get current hotspot/WiFi IP
echo [1/3] Detecting your current PC IP address...
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "192.168.56"') do (
    set IP=%%A
    goto :found
)

:found
:: Remove leading space from IP
set IP=%IP: =%
echo     Your PC IP: %IP%
echo.

echo [2/3] Starting Flask backend on %IP%:5001 ...
echo     (Keep this window open while using the app)
echo.

:: Update Flutter constants.dart with current IP
set CONST_FILE=..\flutterapp\lib\utils\constants.dart
if exist %CONST_FILE% (
    powershell -Command "(Get-Content '%CONST_FILE%') -replace \"static const String baseUrl = 'http://[^']+'\", \"static const String baseUrl = 'http://%IP%:5001'; // Auto-detected IP\" | Set-Content '%CONST_FILE%'"
    echo [INFO] Updated constants.dart with IP: %IP%
    echo [INFO] Remember to rebuild the APK: flutter build apk --release
    echo.
)

:: Start Flask server
python app.py

pause
