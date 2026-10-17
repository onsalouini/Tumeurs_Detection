@echo off
REM Tumor Testing Studio - Script de démarrage du Backend API

cd /d %~dp0

REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo Erreur: Impossible d'activer l'environnement virtuel
    echo Créer d'abord l'environnement avec: python -m venv .venv
    pause
    exit /b 1
)

REM Naviguer au répertoire app
cd app

REM Lancer l'API FastAPI
echo.
echo ==========================================
echo  🚀 Démarrage du Backend API...
echo ==========================================
echo.
echo L'API sera disponible sur: http://127.0.0.1:8000
echo Documentation Swagger: http://127.0.0.1:8000/docs
echo Documentation ReDoc: http://127.0.0.1:8000/redoc
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
