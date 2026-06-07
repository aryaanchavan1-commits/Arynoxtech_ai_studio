@echo off
title Arynox AI News Studio
echo ============================================
echo    Arynox AI News Studio
echo ============================================
echo.

:: Check if venv exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found.
    echo Run setup.bat first to install dependencies.
    pause
    exit /b 1
)

:: Activate and run
call venv\Scripts\activate.bat
streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App failed to start. Run setup.bat and try again.
    pause
)
