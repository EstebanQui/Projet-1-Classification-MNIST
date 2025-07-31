# 🤖 Projet 1: Classification MNIST

<div align="center">

![MNIST](https://img.shields.io/badge/Dataset-MNIST-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![ONNX](https://img.shields.io/badge/Export-ONNX-green)
![JavaScript](https://img.shields.io/badge/Web-JavaScript-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

**Intelligence Artificielle de classification de chiffres manuscrits avec interface web interactive**

> **📝 Note** : Ce projet utilise maintenant `MNIST_Training_clean.ipynb` - version optimisée avec hyperparamètres améliorés pour une meilleure accuracy.

[🚀 Démo Live](https://estebanqui.github.io/Projet-1-Classification-MNIST/index.html)

</div>

## 📋 Vue d'ensemble

Ce projet implémente une **Intelligence Artificielle complète** pour la classification de chiffres manuscrits MNIST, de l'entraînement à l'intégration web. L'objectif est de créer un système de reconnaissance de chiffres accessible via navigateur web.

### ✨ Fonctionnalités principales

- 🧠 **Modèle CNN moderne** : Architecture convolutionnelle optimisée
- 🎯 **Précision >98%** : Performance state-of-the-art sur MNIST (batch size optimisé)  
- 🌐 **Interface web interactive** : Dessinez et obtenez des prédictions en temps réel
- ⚡ **Inférence côté client** : Aucun serveur requis grâce à ONNX.js
- 📱 **Design responsive** : Compatible mobile et desktop
- 📊 **Visualisations complètes** : Métriques d'entraînement et analyses

## 🏗️ Architecture du projet

```
Projet_1_Classification_MNIST/
├── 📓 MNIST_Training_clean.ipynb     # Notebook principal (entraînement complet)
├── 🌐 web/
│   ├── index.html              # Interface web interactive
│   └── mnist_model.onnx        # Modèle exporté (3.3MB)
├── 🤖 models/
│   ├── mnist_model_best.pth    # Meilleur modèle PyTorch
│   └── mnist_model.onnx        # Modèle ONNX pour le web
├── 📁 data/                    # Données MNIST (auto-téléchargées)
├── 🐍 scripts/                 # Scripts utilitaires
├── 📝 README.md               # Documentation
└── 🚫 .gitignore              # Exclusions Git
```

## 🎯 Performances

| Métrique | Valeur | Status |
|----------|--------|---------|
| **Accuracy Test** | **>98%** | ✅ |
| **Temps d'entraînement** | ~5 minutes | ⚡ |
| **Taille modèle ONNX** | 3.3 MB | 📦 |
| **Inférence web** | <100ms | 🚀 |
| **Paramètres modèle** | ~817K | 🧠 |

## 🚀 Installation et utilisation

### 1️⃣ Prérequis

```bash
# Python 3.8+ avec les packages
pip install torch torchvision numpy matplotlib requests jupyter
pip install onnxruntime  # Optionnel pour tests ONNX
pip install tensorboard  # Pour les visualisations avancées
```

### 2️⃣ Entraînement du modèle

**Option A: Jupyter Notebook (Recommandé)**
```bash
jupyter notebook MNIST_Training_clean.ipynb
# Exécutez toutes les cellules séquentiellement
```

**Option B: Scripts Python**
```bash
cd scripts
python train_mnist_working.py
python export_onnx_simple.py
```

### 3️⃣ Interface web

**Lancement automatique via notebook:**
- La dernière cellule du notebook lance automatiquement l'interface
- Cliquez sur le lien généré : `http://localhost:8000/`

**Lancement manuel:**
```bash
# Depuis la racine du projet
python -m http.server 8000
# Ouvrez: http://localhost:8000/
```

## 🌐 Interface web

### 🎨 Utilisation

1. **Dessinez un chiffre** (0-9) dans le canvas blanc avec votre souris/doigt
2. **La prédiction apparaît automatiquement** en temps réel
3. **Confidence score** affiché pour chaque classe
4. **Effacez** avec le bouton 🗑️ pour tester d'autres chiffres

### 📱 Fonctionnalités

- ✅ **Prédiction temps réel** : Pas de bouton nécessaire
- ✅ **Compatible tactile** : Fonctionne sur mobile/tablette  
- ✅ **Design moderne** : Interface intuitive et responsive
- ✅ **Offline après chargement** : Aucun serveur externe requis

## 📊 Visualisation avec TensorBoard

### 🚀 Remplacement de Matplotlib

Ce projet intègre maintenant **TensorBoard** pour remplacer les graphiques matplotlib statiques par une interface web interactive et en temps réel.

### 📈 Avantages de TensorBoard

- 🔄 **Temps réel** : Visualisation en direct pendant l'entraînement
- 🎛️ **Interactivité** : Zoom, filtrage, comparaison d'expériences
- 📊 **Métriques avancées** : Histogrammes, distributions, gradients
- 🌐 **Interface web** : Accessible depuis n'importe quel navigateur
- 📱 **Responsive** : Fonctionne sur desktop et mobile

### 🛠️ Utilisation

#### 1. Entraînement avec TensorBoard
```bash
python scripts/mnist_tensorboard_training.py
```

#### 2. Lancement de TensorBoard
```bash
# Option A : Script Windows optimisé (recommandé)
python scripts/launch_tensorboard_windows.py

# Option B : Script automatique
python scripts/launch_tensorboard.py

# Option C : Commande manuelle
python -m tensorboard.main --logdir=runs --port=6006
```

#### 3. Test de l'intégration
```bash
python scripts/test_tensorboard.py
```

### 📋 Métriques disponibles

- **Loss d'entraînement** : Évolution de la perte par batch et par époque
- **Accuracy d'entraînement** : Précision sur les données d'entraînement
- **Loss de test** : Évolution de la perte sur les données de test
- **Accuracy de test** : Précision sur les données de test
- **Différence Train-Test** : Détection automatique du surapprentissage
- **Histogrammes des paramètres** : Distribution des poids et biais
- **Distribution des gradients** : Analyse de l'optimisation

### 📁 Structure des logs

```
runs/
├── mnist_experiment_20241201_143022/
│   ├── events.out.tfevents.1701430222.hostname
│   └── ...
└── ...
```

### 🔧 Configuration

Le fichier `scripts/config_tensorboard.py` permet de personnaliser :
- Intervalles de logging
- Métriques à afficher
- Paramètres du serveur
- Options de visualisation

### 📚 Documentation

Consultez `docs/TENSORBOARD_GUIDE.md` pour un guide complet d'utilisation.

## 🤖 Modèle CNN

### 🏗️ Architecture

```python
MNISTNet(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)    # 28×28 → 28×28
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)   # 14×14 → 14×14
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)  # 7×7 → 7×7
  (pool): MaxPool2d(2, 2)                             # Réduction 2×2
  (fc1): Linear(1152, 512)                            # Fully connected
  (fc2): Linear(512, 256)                             
  (fc3): Linear(256, 10)                              # 10 classes de sortie
  (dropout): Dropout(0.5)                             # Régularisation
)
```

### ⚙️ Hyperparamètres

- **Optimizer** : Adam (lr=0.001)
- **Loss function** : CrossEntropyLoss
- **Batch size** : 128
- **Epochs** : 10 
- **Dropout** : 0.5
- **Data augmentation** : Normalisation MNIST standard

## 📊 Résultats d'entraînement

Le notebook génère automatiquement :

- 📈 **Courbes de loss** (train/test)
- 📊 **Évolution de l'accuracy** par époque
- 🎯 **Métriques de performance** détaillées
- 🔍 **Exemples de prédictions** sur le dataset de test

## 🛠️ Technologies utilisées

| Composant | Technologie | Version |
|-----------|-------------|---------|
| **ML Framework** | PyTorch | 2.x |
| **Export Model** | ONNX | 1.11 |
| **Web Runtime** | ONNX.js | Latest |
| **Frontend** | Vanilla JS | ES6+ |
| **Styling** | CSS3 | Modern |
| **Data Viz** | Matplotlib + TensorBoard | 3.x + Latest |

## 📁 Fichiers importants

| Fichier | Description | Taille |
|---------|-------------|--------|
| `MNIST_Training_clean.ipynb` | 📓 Notebook principal complet | ~42KB |
| `web/index.html` | 🌐 Interface web interactive | ~17KB |
| `web/mnist_model.onnx` | 🤖 Modèle exporté pour le web | 3.3MB |
| `models/mnist_model_best.pth` | 💾 Meilleur modèle PyTorch | 9.8MB |
| `scripts/mnist_tensorboard_training.py` | 🚀 Entraînement avec TensorBoard | ~15KB |
| `scripts/launch_tensorboard.py` | 📊 Lanceur TensorBoard automatique | ~8KB |
| `scripts/launch_tensorboard_windows.py` | 🪟 Lanceur Windows optimisé | ~12KB |
| `docs/TENSORBOARD_GUIDE.md` | 📚 Guide complet TensorBoard | ~12KB |
| `docs/TROUBLESHOOTING.md` | 🔧 Guide de dépannage | ~8KB |

## 🔧 Développement

### 🐛 Debug et tests

```bash
# Tester l'export ONNX
python -c "import onnxruntime; print('ONNX Runtime OK')"

# Vérifier la structure du modèle
python -c "import torch; m = torch.load('models/mnist_model_best.pth', map_location='cpu'); print('Model loaded')"

# Lancer un serveur web simple
python -m http.server 8000 --directory web
```

### 🔄 Modifications du modèle

1. Modifiez l'architecture dans `MNISTNet` (notebook cellule 2)
2. Réentraînez le modèle (cellules 8+)
3. Exportez vers ONNX (cellule 5)
4. Testez l'interface web

## 🎉 Résultats

✅ **Objectifs atteints :**
- Précision >98% sur le dataset de test MNIST (grâce au batch size optimisé)
- Interface web fonctionnelle avec prédictions temps réel
- Export ONNX réussi pour déploiement navigateur
- Code bien documenté et reproductible

## 📚 Ressources

- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/) - Dataset officiel
- [PyTorch Documentation](https://pytorch.org/docs/) - Framework ML
- [ONNX.js](https://onnxjs.org/) - Runtime JavaScript
- [Architecture CNN](https://cs231n.github.io/convolutional-networks/) - Théorie

## 🤝 Contribution

Les contributions sont les bienvenues ! Ouvrez une issue ou proposez une pull request.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

<div align="center">

**🎯 Projet réalisé avec passion pour l'apprentissage automatique**

⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile !

</div> 