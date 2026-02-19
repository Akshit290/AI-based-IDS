@echo off
REM ============================================
REM Network Intrusion Detection System Startup
REM ============================================

setlocal enabledelayedexpansion

cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║   AI-Based Network Intrusion Detection System (NIDS)       ║
echo ║   Production Ready v1.0.0                                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment found
)

REM Activate virtual environment
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Install dependencies
echo.
echo [3/4] Checking dependencies...
pip install -q -r requirements.txt > nul 2>&1
if !errorlevel! neq 0 (
    echo ⚠️  Installing dependencies (this may take a few minutes)...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)
echo ✅ Dependencies installed

REM Generate sample data if not exists
if not exist "data\network_traffic.csv" (
    echo.
    echo [4/4] Generating sample data...
    python src\data_Pipelines\generate_sample_data.py
    if !errorlevel! neq 0 (
        echo ⚠️  Could not generate sample data
    ) else (
        echo ✅ Sample data generated
    )
) else (
    echo [4/4] ✅ Sample data already exists
)

REM Show menu
:menu
cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         Network Intrusion Detection System Menu            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo 1. Generate Sample Data
echo 2. Train Random Forest Model
echo 3. Train Gradient Boosting Model
echo 4. Train Ensemble Model
echo 5. Run API Server
echo 6. Launch Dashboard
echo 7. Run Tests
echo 8. View Documentation
echo 9. Exit
echo.
set /p choice="Choose an option (1-9): "

if "%choice%"=="1" (
    echo Generating sample data...
    python main.py generate-data --samples 10000
    pause
    goto menu
)

if "%choice%"=="2" (
    echo Training Random Forest model...
    python main.py train --model random_forest --data data/network_traffic.csv --save
    pause
    goto menu
)

if "%choice%"=="3" (
    echo Training Gradient Boosting model...
    python main.py train --model gradient_boosting --data data/network_traffic.csv --save
    pause
    goto menu
)

if "%choice%"=="4" (
    echo Training Ensemble model...
    python main.py train --model ensemble --data data/network_traffic.csv --save
    pause
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo Starting API Server...
    echo.
    echo 🚀 API running at http://localhost:5000
    echo 📚 API documentation: http://localhost:5000/api/v1/
    echo 🛑 Press Ctrl+C to stop
    echo.
    python main.py api --port 5000
    goto menu
)

if "%choice%"=="6" (
    echo.
    echo Starting Dashboard...
    echo.
    echo 📊 Dashboard running at http://localhost:8050
    echo 🛑 Press Ctrl+C to stop
    echo.
    python main.py dashboard --port 8050
    goto menu
)

if "%choice%"=="7" (
    echo Running tests...
    pytest tests/ -v
    pause
    goto menu
)

if "%choice%"=="8" (
    echo.
    echo Available documentation:
    echo - README.md: Project overview
    echo - QUICKSTART.md: 5-minute guide
    echo - ARCHITECTURE.md: System design
    echo - docs/api.md: API reference
    echo - FILE_INVENTORY.md: File listing
    echo.
    pause
    goto menu
)

if "%choice%"=="9" (
    echo Goodbye!
    exit /b 0
)

echo Invalid option. Please try again.
pause
goto menu
