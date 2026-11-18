@echo off
cd /d "%~dp0"
title PathSnap Backend - Laptop 1 (Model Server)
echo ==============================================================
echo          PATHSNAP: LAPTOP 1 BACKEND (MODEL & API SERVER)
echo ==============================================================
echo.
echo Checking IP Configuration...
ipconfig | findstr /i "IPv4"
echo.
echo Starting FastAPI Server on 0.0.0.0:8000...
echo Keep this terminal window open and visible for the evaluator!
echo.
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
pause
