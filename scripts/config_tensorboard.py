#!/usr/bin/env python3
"""
Configuration pour l'intégration TensorBoard dans le projet MNIST
"""

import os
from datetime import datetime

class TensorBoardConfig:
    """
    Configuration centralisée pour TensorBoard
    """
    
    # Dossiers
    LOG_DIR = "runs"
    MODELS_DIR = "models"
    DATA_DIR = "data"
    
    # Paramètres d'entraînement
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    
    # Paramètres TensorBoard
    LOG_INTERVAL = 50  # Log tous les N batches
    GRADIENT_LOG_INTERVAL = 50  # Log des gradients tous les N batches
    PARAMETER_LOG_INTERVAL = 1  # Log des paramètres chaque époque
    
    # Métriques à logger
    METRICS_TO_LOG = {
        'scalars': [
            'Training/Batch_Loss',
            'Training/Batch_Accuracy', 
            'Epoch/Train_Loss',
            'Epoch/Train_Accuracy',
            'Epoch/Test_Loss',
            'Epoch/Test_Accuracy',
            'Epoch/Train_Test_Accuracy_Diff'
        ],
        'histograms': [
            'Parameters',
            'Gradients'
        ],
        'text': [
            'Training_Summary'
        ]
    }
    
    @classmethod
    def get_experiment_name(cls, prefix="mnist_experiment"):
        """
        Génère un nom d'expérience unique
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}"
    
    @classmethod
    def get_log_dir(cls, experiment_name=None):
        """
        Retourne le chemin complet du dossier de logs
        """
        if experiment_name is None:
            experiment_name = cls.get_experiment_name()
        
        return os.path.join(cls.LOG_DIR, experiment_name)
    
    @classmethod
    def create_directories(cls):
        """
        Crée tous les dossiers nécessaires
        """
        directories = [cls.LOG_DIR, cls.MODELS_DIR, cls.DATA_DIR]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Dossier créé/vérifié: {directory}")
    
    @classmethod
    def get_model_path(cls, model_name, use_tensorboard=True):
        """
        Retourne le chemin complet pour sauvegarder un modèle
        """
        suffix = "_tensorboard" if use_tensorboard else ""
        filename = f"{model_name}{suffix}.pth"
        return os.path.join(cls.MODELS_DIR, filename)

# Configuration pour les visualisations
class VisualizationConfig:
    """
    Configuration pour les visualisations TensorBoard
    """
    
    # Couleurs pour les graphiques
    COLORS = {
        'train': '#1f77b4',  # Bleu
        'test': '#ff7f0e',   # Orange
        'loss': '#d62728',   # Rouge
        'accuracy': '#2ca02c'  # Vert
    }
    
    # Intervalles de mise à jour
    UPDATE_INTERVALS = {
        'scalars': 1,      # Mise à jour chaque step
        'histograms': 10,   # Mise à jour tous les 10 steps
        'images': 100,      # Mise à jour tous les 100 steps
        'text': 1          # Mise à jour chaque époque
    }
    
    # Limites pour les graphiques
    LIMITS = {
        'accuracy_min': 0,
        'accuracy_max': 100,
        'loss_min': 0,
        'loss_max': 5
    }

# Configuration pour le logging avancé
class AdvancedLoggingConfig:
    """
    Configuration pour le logging avancé
    """
    
    # Logging des gradients
    LOG_GRADIENTS = True
    GRADIENT_CLIP_NORM = 1.0
    
    # Logging des paramètres
    LOG_PARAMETERS = True
    PARAMETER_STATS = ['mean', 'std', 'min', 'max']
    
    # Logging des images
    LOG_IMAGES = True
    IMAGE_LOG_INTERVAL = 100  # Log d'images tous les N batches
    MAX_IMAGES_PER_LOG = 16
    
    # Logging de l'architecture
    LOG_MODEL_GRAPH = True
    
    # Logging des métriques personnalisées
    CUSTOM_METRICS = {
        'learning_rate': True,
        'gradient_norm': True,
        'parameter_norm': True
    }

# Configuration pour le lancement de TensorBoard
class TensorBoardLaunchConfig:
    """
    Configuration pour le lancement de TensorBoard
    """
    
    # Paramètres du serveur
    DEFAULT_PORT = 6006
    HOST = "localhost"
    
    # Options de lancement
    AUTO_OPEN_BROWSER = True
    RELOAD_INTERVAL = 5  # Secondes
    
    # Configuration du serveur
    SERVER_CONFIG = {
        'bind_all': False,
        'max_reload_threads': 1,
        'reload_task': 'auto'
    }

# Configuration pour les tests
class TestConfig:
    """
    Configuration pour les tests TensorBoard
    """
    
    # Paramètres de test
    TEST_EPOCHS = 5
    TEST_BATCHES = 10
    TEST_DATA_SIZE = 1000
    
    # Validation
    VALIDATE_LOGS = True
    CHECK_FILE_SIZES = True
    MIN_LOG_SIZE = 1024  # bytes

# Configuration globale
class GlobalConfig:
    """
    Configuration globale combinant toutes les configurations
    """
    
    # Références aux configurations
    TB = TensorBoardConfig
    VIS = VisualizationConfig
    ADV = AdvancedLoggingConfig
    LAUNCH = TensorBoardLaunchConfig
    TEST = TestConfig
    
    # Paramètres globaux
    DEBUG = False
    VERBOSE = True
    
    @classmethod
    def print_config(cls):
        """
        Affiche la configuration actuelle
        """
        print("🔧 CONFIGURATION TENSORBOARD")
        print("=" * 50)
        
        print(f"📁 Dossiers:")
        print(f"   Logs: {cls.TB.LOG_DIR}")
        print(f"   Modèles: {cls.TB.MODELS_DIR}")
        print(f"   Données: {cls.TB.DATA_DIR}")
        
        print(f"\n⚙️ Paramètres d'entraînement:")
        print(f"   Époques: {cls.TB.NUM_EPOCHS}")
        print(f"   Learning rate: {cls.TB.LEARNING_RATE}")
        print(f"   Batch size: {cls.TB.BATCH_SIZE}")
        
        print(f"\n📊 Logging:")
        print(f"   Interval batch: {cls.TB.LOG_INTERVAL}")
        print(f"   Interval gradients: {cls.TB.GRADIENT_LOG_INTERVAL}")
        print(f"   Log gradients: {cls.ADV.LOG_GRADIENTS}")
        print(f"   Log paramètres: {cls.ADV.LOG_PARAMETERS}")
        print(f"   Log images: {cls.ADV.LOG_IMAGES}")
        
        print(f"\n🌐 Serveur:")
        print(f"   Port: {cls.LAUNCH.DEFAULT_PORT}")
        print(f"   Auto-open: {cls.LAUNCH.AUTO_OPEN_BROWSER}")
        
        print("=" * 50)

# Fonction utilitaire pour créer une configuration personnalisée
def create_custom_config(**kwargs):
    """
    Crée une configuration personnalisée
    
    Args:
        **kwargs: Paramètres à personnaliser
        
    Returns:
        dict: Configuration personnalisée
    """
    config = {
        'num_epochs': TensorBoardConfig.NUM_EPOCHS,
        'learning_rate': TensorBoardConfig.LEARNING_RATE,
        'batch_size': TensorBoardConfig.BATCH_SIZE,
        'log_interval': TensorBoardConfig.LOG_INTERVAL,
        'log_gradients': AdvancedLoggingConfig.LOG_GRADIENTS,
        'log_parameters': AdvancedLoggingConfig.LOG_PARAMETERS,
        'log_images': AdvancedLoggingConfig.LOG_IMAGES,
        'port': TensorBoardLaunchConfig.DEFAULT_PORT
    }
    
    # Mettre à jour avec les paramètres personnalisés
    config.update(kwargs)
    
    return config

if __name__ == "__main__":
    # Afficher la configuration par défaut
    GlobalConfig.print_config()
    
    # Exemple de configuration personnalisée
    print("\n🎛️ Exemple de configuration personnalisée:")
    custom_config = create_custom_config(
        num_epochs=15,
        learning_rate=0.0005,
        batch_size=64,
        log_interval=25
    )
    
    for key, value in custom_config.items():
        print(f"   {key}: {value}") 