@echo off
REM Tumor Testing Studio - Script de démarrage du Frontend Streamlit

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

REM Naviguer au répertoire streamlit
cd streamlit

REM Lancer Streamlit
echo.
echo ==========================================
echo  🎨 Démarrage du Frontend Streamlit...
echo ==========================================
echo.
echo L'application web sera disponible sur: http://localhost:8501
echo Assurez-vous que le Backend API tourne sur: http://127.0.0.1:8000
echo.

python -m streamlit run app.py

pause
