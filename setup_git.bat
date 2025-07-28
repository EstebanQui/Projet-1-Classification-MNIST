@echo off
echo ================================================
echo    Configuration Git pour Projet MNIST
echo ================================================
echo.

REM Vérifier si Git est installé
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git n'est pas installé !
    echo.
    echo 📥 Veuillez installer Git depuis : https://git-scm.com/download/win
    echo 🔄 Puis relancez ce script
    echo.
    pause
    exit /b 1
)

echo ✅ Git détecté !
echo.

REM Configuration Git (remplacez par vos informations)
echo 🔧 Configuration de Git...
git config --global user.name "EstebanQui"
git config --global user.email "votre-email@example.com"

REM Initialiser le repository
echo 🚀 Initialisation du repository Git...
git init

REM Ajouter le remote origin
echo 🌐 Ajout du remote GitHub...
git remote add origin https://github.com/EstebanQui/Projet-1-Classification-MNIST.git

REM Ajouter tous les fichiers
echo 📁 Ajout des fichiers au staging...
git add .

REM Premier commit
echo 💾 Premier commit...
git commit -m "🎉 Initial commit: Projet MNIST Classification

✅ Modèle CNN PyTorch entraîné
✅ Interface web interactive  
✅ Export ONNX pour inférence browser
✅ Jupyter Notebook complet
✅ Documentation et scripts"

REM Pousser vers GitHub
echo 🚀 Push vers GitHub...
git branch -M main
git push -u origin main

echo.
echo 🎉 SUCCÈS ! Projet mis sur GitHub :
echo 🌐 https://github.com/EstebanQui/Projet-1-Classification-MNIST
echo.
pause 