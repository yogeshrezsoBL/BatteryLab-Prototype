@echo off
title BatteryLab Launcher (Windows, py)

REM Prefer Python 3.11 (best compatibility), else 3.10, else default
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo Python launcher 'py' not found. Please install Python from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

echo Trying Python 3.11...
py -3.11 --version >nul 2>nul
if %errorlevel% equ 0 (
    set PYCMD=py -3.11
) else (
    echo Python 3.11 not found. Trying Python 3.10...
    py -3.10 --version >nul 2>nul
    if %errorlevel% equ 0 (
        set PYCMD=py -3.10
    ) else (
        echo Falling back to default 'py'...
        set PYCMD=py
    )
)

echo Creating virtual environment...
%PYCMD% -m venv .venv
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Failed to create virtual environment. Please ensure Python is installed correctly.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip >nul

echo Installing requirements (this may take a minute)...
python -m pip install -r requirements.txt

echo Launching BatteryLab in your browser...
streamlit run app.py

echo.
echo If Streamlit failed with Python 3.13, install Python 3.11 (recommended) from python.org and re-run this launcher.
pause
