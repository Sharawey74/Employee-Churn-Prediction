@echo off
echo Starting Employee Churn Prediction GUI...
echo.

REM Check if we're in the right directory
if not exist app.py (
    echo Error: app.py not found!
    echo Please run this script from the GUI directory.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Streamlit not found. Installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install requirements.
        pause
        exit /b 1
    )
)

echo Launching Streamlit application...
echo.
echo The application will open in your default web browser.
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run app.py

pause
