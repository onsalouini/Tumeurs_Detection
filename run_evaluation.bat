@echo off
REM Tumor Testing Studio - Script d'évaluation du modèle

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

REM Vérifier que l'API est disponible
echo.
echo ==========================================
echo  ⚠️  Vérification de l'API...
echo ==========================================
echo.
echo Assurez-vous que le Backend API tourne sur: http://127.0.0.1:8000
echo Si ce n'est pas le cas, exécutez d'abord: run_backend.bat
echo.
timeout /t 3

REM Lancer l'évaluation
echo.
echo ==========================================
echo  📊 Démarrage de l'évaluation...
echo ==========================================
echo.

python evaluate.py --data-dir ../data_backup/Testing --recursive --output eval_report.json

echo.
echo ==========================================
echo  ✅ Évaluation terminée!
echo ==========================================
echo Résultats: eval_report.json

pause
