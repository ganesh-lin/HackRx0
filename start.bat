@echo off
REM HackRX API Startup Script for Windows

echo Starting HackRX Policy Analysis API...

REM Check if .env file exists
if not exist ".env" (
    echo Error: .env file not found. Please create one with your configuration.
    exit /b 1
)

echo Environment file found ✓

REM Create logs directory
if not exist "logs" mkdir logs

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -r requirements.txt

echo Starting the API server...
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
