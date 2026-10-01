@echo off
chcp 65001 > nul
echo ================================================================================
echo   CONQUER AI: MLOps Reliability ^& Survival Lifecycle Engine (GP-071)
echo   One-Click Automated Ingestion, Analytics, Pytest Suite ^& Latency Benchmarks
echo ================================================================================
echo.

echo [1/4] Generating Calibrated Stochastic Domain Telemetry (50,000 records)...
python src/data_generator.py --records 50000
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data generation phase failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Executing Executive Rich TUI ^& Analytical Degradation Engine...
python src/interface.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Executive interface execution failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Running Automated Pytest Mathematical Invariant Suite...
python -m pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pytest invariant suite failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Executing Quantitative Latency Profiler (30 iterations, p50/p95 SLAs)...
python tests/benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Benchmark latency SLA failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================================
echo   [SUCCESS] Full MLOps Reliability Engine Executed Successfully!
echo   All mathematical invariants, DIP mocks, and latency SLAs verified.
echo ================================================================================