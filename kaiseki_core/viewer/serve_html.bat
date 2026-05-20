@echo off
cd /d "%~dp0\..\.."
echo [P/H] Killing Streamlit on 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do taskkill /F /PID %%a 2^>nul
timeout /t 1 /nobreak >nul
echo [P/H] Generating dashboard...
python -m kaiseki_core.viewer.html_export
if errorlevel 1 (
    echo Failed.
    pause
    exit /b 1
)
echo [P/H] Starting http.server on 8501...
python -m http.server 8501 --directory kaiseki_core\viewer\static
