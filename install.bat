@echo off
REM Tumor Testing Studio - Script d'installation

echo.
echo ==========================================
echo  📦 Installation du Projet
echo ==========================================
echo.

REM Vérifier Python
echo Vérification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Erreur: Python n'est pas installé ou pas dans le PATH
    echo Visitez: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python trouvé

REM Créer l'environnement virtuel s'il n'existe pas
if not exist ".venv" (
    echo.
    echo 📁 Création de l'environnement virtuel...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Erreur lors de la création de l'environnement virtuel
        pause
        exit /b 1
    )
    echo ✅ Environnement virtuel créé
)

REM Activer l'environnement virtuel
echo.
echo 🔧 Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Erreur lors de l'activation de l'environnement virtuel
    pause
    exit /b 1
)
echo ✅ Environnement virtuel activé

REM Mettre à jour pip
echo.
echo 📦 Mise à jour de pip...
python -m pip install --upgrade pip --quiet

REM Installer les dépendances
echo.
echo 📥 Installation des dépendances...
echo Cela peut prendre 10-15 minutes (TensorFlow est volumineux)...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Erreur lors de l'installation des dépendances
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  ✅ Installation terminée avec succès!
echo ==========================================
echo.
echo Prochaines étapes:
echo 1. Lancer le Backend:    run_backend.bat
echo 2. Lancer le Frontend:   run_frontend.bat
echo 3. Évaluer le modèle:    run_evaluation.bat
echo.
echo Documentation: README.md
echo.

pause
