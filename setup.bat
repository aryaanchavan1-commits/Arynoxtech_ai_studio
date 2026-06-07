@echo off
title AI News Studio - Setup
echo ============================================
echo    AI News Studio - Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found: 
python --version

:: Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo [OK] Virtual environment exists
)

:: Activate venv and install packages
echo Installing dependencies...
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip -q

:: Install requirements
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [WARNING] Some packages may have failed. Check above for errors.
) else (
    echo [OK] Dependencies installed
)

:: Download Wav2Lip model
echo.
echo Downloading Wav2Lip model...
if not exist "models\wav2lip_gan.pth" (
    mkdir models >nul 2>&1
    python -c "
import requests, sys
url = 'https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth'
print('Downloading from GitHub... (~150MB)')
try:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open('models/wav2lip_gan.pth', 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f'\rDownloading... {pct}%', end='')
    print('\n[OK] Model downloaded successfully')
except Exception as e:
    print(f'GitHub download failed: {e}')
    print('Trying alternative source (HuggingFace)...')
    url2 = 'https://huggingface.co/sparksammy/wav2lip-gan/resolve/main/wav2lip_gan.pth'
    try:
        resp2 = requests.get(url2, stream=True, timeout=120)
        resp2.raise_for_status()
        with open('models/wav2lip_gan.pth', 'wb') as f:
            for chunk in resp2.iter_content(chunk_size=8192):
                f.write(chunk)
        print('[OK] Model downloaded from HuggingFace')
    except Exception as e2:
        print(f'All download sources failed: {e2}')
        print('Please manually download from:')
        print('  https://github.com/Rudrabha/Wav2Lip/releases')
        print('  and place wav2lip_gan.pth in models/')
"
) else (
    echo [OK] Model already exists
)

:: Check FFmpeg
echo.
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg not found. Required for video processing.
    echo Install: winget install ffmpeg
    echo Or download from: https://ffmpeg.org/download.html
) else (
    echo [OK] FFmpeg found
)

:: Setup .env if needed
if not exist ".env" (
    echo.
    echo Creating .env file from template...
    copy .env .env >nul
    echo [INFO] Edit .env to add your Groq API key
) else (
    echo [OK] .env file exists
)

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo To run the app:
echo   1. Edit .env with your Groq API key
echo   2. Run: streamlit run app.py
echo.
echo Get a Groq API key: https://console.groq.com
echo.
pause
