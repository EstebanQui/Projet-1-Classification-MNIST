# 🚀 Instructions post-installation Git

## ✅ Une fois Git installé et votre PowerShell redémarré :

### 1️⃣ Vérifier Git
```bash
git --version
```

### 2️⃣ Configurer Git (remplacez par votre email GitHub !)
```bash
git config --global user.name "EstebanQui"
git config --global user.email "VOTRE-EMAIL@example.com"
```

### 3️⃣ Lancer l'automatisation
```bash
.\setup_git.bat
```

### 4️⃣ Ou commandes manuelles :
```bash
# Initialiser le repository
git init

# Ajouter le remote GitHub
git remote add origin https://github.com/EstebanQui/Projet-1-Classification-MNIST.git

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "🎉 Initial commit: Projet MNIST Classification"

# Push vers GitHub
git branch -M main
git push -u origin main
```

## 🎯 Résultat attendu :
```
🎉 SUCCÈS ! Projet mis sur GitHub :
🌐 https://github.com/EstebanQui/Projet-1-Classification-MNIST
``` 