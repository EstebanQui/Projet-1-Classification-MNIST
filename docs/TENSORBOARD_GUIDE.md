# Guide TensorBoard pour le Projet MNIST

## 🎯 Objectif

Ce guide explique comment utiliser **TensorBoard** pour remplacer les graphiques matplotlib dans le projet de classification MNIST. TensorBoard offre une interface web interactive et plus avancée pour visualiser l'entraînement des modèles de deep learning.

## 📋 Prérequis

### Installation de TensorBoard

```bash
pip install tensorboard
```

### Vérification de l'installation

```python
import tensorboard
print("TensorBoard installé avec succès!")
```

## 🚀 Utilisation

### 1. Entraînement avec TensorBoard

Exécutez le script d'entraînement avec TensorBoard :

```bash
python scripts/mnist_tensorboard_training.py
```

Ce script va :
- Entraîner le modèle CNN sur MNIST
- Générer des logs TensorBoard dans le dossier `runs/`
- Sauvegarder les modèles dans `models/`

### 2. Lancement de TensorBoard

#### Option A : Script automatique (Recommandé)

```bash
python scripts/launch_tensorboard.py
```

Ce script va :
- Vérifier l'installation de TensorBoard
- Lister les expériences disponibles
- Lancer TensorBoard automatiquement
- Ouvrir le navigateur

#### Option B : Commande manuelle

```bash
tensorboard --logdir=runs --port=6006
```

Puis ouvrez http://localhost:6006 dans votre navigateur.

## 📊 Fonctionnalités TensorBoard

### Métriques en Temps Réel

- **Loss d'entraînement** : Évolution de la perte pendant l'entraînement
- **Accuracy d'entraînement** : Précision sur les données d'entraînement
- **Loss de test** : Évolution de la perte sur les données de test
- **Accuracy de test** : Précision sur les données de test
- **Différence Train-Test** : Détection du surapprentissage

### Visualisations Avancées

#### 1. Graphiques Scalaires
- Courbes d'évolution de la loss et accuracy
- Comparaison train vs test
- Détection de l'overfitting

#### 2. Histogrammes
- **Paramètres du modèle** : Distribution des poids et biais
- **Gradients** : Distribution des gradients pendant l'entraînement

#### 3. Textes
- **Résumé d'entraînement** : Métriques finales et configuration

## 🔍 Interface TensorBoard

### Onglets Principaux

1. **SCALARS** : Graphiques de métriques (loss, accuracy)
2. **IMAGES** : Visualisation d'images (si configuré)
3. **GRAPHS** : Architecture du modèle (si configuré)
4. **DISTRIBUTIONS** : Histogrammes des paramètres
5. **HISTOGRAMS** : Distribution des gradients
6. **TEXT** : Résumés textuels

### Fonctionnalités Interactives

- **Zoom et défilement** sur les graphiques
- **Filtrage** par métrique
- **Comparaison** entre expériences
- **Export** des graphiques
- **Actualisation** en temps réel

## 📁 Structure des Logs

```
runs/
├── mnist_experiment_20241201_143022/
│   ├── events.out.tfevents.1701430222.hostname
│   └── ...
├── mnist_experiment_20241201_150145/
│   ├── events.out.tfevents.1701432585.hostname
│   └── ...
└── ...
```

## 🔧 Configuration Avancée

### Personnalisation des Logs

```python
from torch.utils.tensorboard import SummaryWriter

# Créer un writer personnalisé
writer = SummaryWriter(
    log_dir="runs/experiment_personnalise",
    comment="description_de_lexperience"
)

# Log de métriques personnalisées
writer.add_scalar('Custom/Metric', value, step)
writer.add_histogram('Custom/Distribution', data, step)
writer.add_text('Custom/Description', text, step)
```

### Logging de Données d'Images

```python
# Log d'images d'exemple
writer.add_images('Training/Examples', images, epoch)
```

### Logging de l'Architecture du Modèle

```python
# Log de l'architecture
dummy_input = torch.randn(1, 1, 28, 28)
writer.add_graph(model, dummy_input)
```

## 🐛 Dépannage

### Problèmes Courants

#### 1. TensorBoard ne démarre pas
```bash
# Vérifier l'installation
pip install tensorboard --upgrade

# Vérifier le port
tensorboard --logdir=runs --port=6007
```

#### 2. Aucun log affiché
- Vérifiez que le dossier `runs/` contient des fichiers
- Assurez-vous que l'entraînement s'est bien terminé
- Vérifiez les permissions des fichiers

#### 3. Erreur de port occupé
```bash
# Utiliser un port différent
tensorboard --logdir=runs --port=6007
```

### Logs de Débogage

```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Comparaison avec Matplotlib

| Fonctionnalité | Matplotlib | TensorBoard |
|----------------|------------|-------------|
| **Temps réel** | ❌ | ✅ |
| **Interactivité** | ❌ | ✅ |
| **Comparaison d'expériences** | ❌ | ✅ |
| **Histogrammes** | ✅ | ✅ |
| **Interface web** | ❌ | ✅ |
| **Facilité d'utilisation** | ✅ | ✅ |
| **Performance** | ✅ | ✅ |

## 🎨 Exemples de Visualisations

### Graphique de Loss
```
Époque 1: Train Loss = 0.5234, Test Loss = 0.5123
Époque 2: Train Loss = 0.2341, Test Loss = 0.2456
...
```

### Graphique d'Accuracy
```
Époque 1: Train Acc = 85.2%, Test Acc = 84.7%
Époque 2: Train Acc = 92.1%, Test Acc = 91.8%
...
```

### Détection d'Overfitting
```
Différence Train-Test > 5% = Possible overfitting
Différence Train-Test < 2% = Bonne généralisation
```

## 🚀 Prochaines Étapes

1. **Expérimenter** avec différents hyperparamètres
2. **Comparer** plusieurs architectures
3. **Analyser** les gradients pour l'optimisation
4. **Exporter** les graphiques pour les rapports
5. **Partager** les résultats avec l'équipe

## 📚 Ressources Additionnelles

- [Documentation officielle TensorBoard](https://www.tensorflow.org/tensorboard)
- [Guide PyTorch TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)
- [Tutoriels TensorBoard](https://www.tensorflow.org/tensorboard/get_started)

---

**Note** : TensorBoard est un outil puissant qui remplace avantageusement les graphiques matplotlib statiques par une interface interactive et en temps réel pour le suivi d'entraînement. 