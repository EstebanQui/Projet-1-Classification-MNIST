#!/usr/bin/env python3
"""
Démonstration rapide de l'intégration TensorBoard
Affiche les principales fonctionnalités et lance un test
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """
    Affiche la bannière de démonstration
    """
    print("🚀 DÉMONSTRATION TENSORBOARD POUR MNIST")
    print("=" * 60)
    print("Ce script démontre l'intégration TensorBoard dans le projet MNIST")
    print("Remplace les graphiques matplotlib par une interface web interactive")
    print("=" * 60)

def check_installation():
    """
    Vérifie l'installation des dépendances
    """
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("-" * 40)
    
    # Vérifier TensorBoard
    try:
        import tensorboard
        print("✅ TensorBoard installé")
    except ImportError:
        print("❌ TensorBoard non installé")
        print("   Installation: pip install tensorboard")
        return False
    
    # Vérifier PyTorch
    try:
        import torch
        print(f"✅ PyTorch installé (version {torch.__version__})")
    except ImportError:
        print("❌ PyTorch non installé")
        print("   Installation: pip install torch torchvision")
        return False
    
    # Vérifier les dossiers
    required_dirs = ["scripts", "models", "data"]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Dossier {directory}/ trouvé")
        else:
            print(f"⚠️ Dossier {directory}/ manquant")
    
    print("-" * 40)
    return True

def run_quick_test():
    """
    Lance un test rapide de TensorBoard
    """
    print("\n🧪 TEST RAPIDE TENSORBOARD")
    print("-" * 40)
    
    # Créer le dossier runs s'il n'existe pas
    os.makedirs("runs", exist_ok=True)
    
    # Lancer le test
    try:
        result = subprocess.run([
            sys.executable, "scripts/test_tensorboard.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test TensorBoard réussi!")
            return True
        else:
            print("❌ Test TensorBoard échoué")
            print(f"Erreur: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test interrompu (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def launch_tensorboard_demo():
    """
    Lance TensorBoard en mode démonstration
    """
    print("\n📊 LANCEMENT TENSORBOARD")
    print("-" * 40)
    
    # Vérifier qu'il y a des logs
    if not os.path.exists("runs"):
        print("❌ Aucun dossier de logs trouvé")
        print("   Exécutez d'abord le test ou l'entraînement")
        return False
    
    log_files = list(Path("runs").glob("*"))
    if not log_files:
        print("❌ Aucun fichier de log trouvé")
        print("   Exécutez d'abord le test ou l'entraînement")
        return False
    
    print("🚀 Lancement de TensorBoard...")
    
    try:
        # Lancer TensorBoard en arrière-plan
        process = subprocess.Popen([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", "runs",
            "--port", "6006"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Attendre le démarrage
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ TensorBoard démarré avec succès!")
            print("🌐 URL: http://localhost:6006")
            
            # Ouvrir le navigateur
            try:
                webbrowser.open("http://localhost:6006")
                print("✅ Navigateur ouvert automatiquement")
            except:
                print("⚠️ Ouvrez manuellement: http://localhost:6006")
            
            print("\n📋 FONCTIONNALITÉS À TESTER:")
            print("1. 📈 Onglet SCALARS: Graphiques de loss et accuracy")
            print("2. 📊 Onglet HISTOGRAMS: Distribution des paramètres")
            print("3. 🖼️ Onglet IMAGES: Exemples d'images (si disponibles)")
            print("4. 📝 Onglet TEXT: Résumés d'entraînement")
            print("5. 🎛️ Interactivité: Zoom, filtrage, comparaison")
            
            print("\n🛑 Pour arrêter TensorBoard: Ctrl+C")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Arrêt de TensorBoard...")
                process.terminate()
                process.wait()
                print("✅ TensorBoard arrêté")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Erreur TensorBoard: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def show_comparison():
    """
    Affiche une comparaison entre Matplotlib et TensorBoard
    """
    print("\n📊 COMPARAISON MATPLOTLIB vs TENSORBOARD")
    print("-" * 50)
    
    comparison = {
        "Temps réel": {"matplotlib": "❌", "tensorboard": "✅"},
        "Interactivité": {"matplotlib": "❌", "tensorboard": "✅"},
        "Comparaison d'expériences": {"matplotlib": "❌", "tensorboard": "✅"},
        "Histogrammes": {"matplotlib": "✅", "tensorboard": "✅"},
        "Interface web": {"matplotlib": "❌", "tensorboard": "✅"},
        "Facilité d'utilisation": {"matplotlib": "✅", "tensorboard": "✅"},
        "Performance": {"matplotlib": "✅", "tensorboard": "✅"},
        "Métriques avancées": {"matplotlib": "⚠️", "tensorboard": "✅"}
    }
    
    print(f"{'Fonctionnalité':<25} {'Matplotlib':<12} {'TensorBoard':<12}")
    print("-" * 50)
    
    for feature, status in comparison.items():
        print(f"{feature:<25} {status['matplotlib']:<12} {status['tensorboard']:<12}")
    
    print("-" * 50)
    print("✅ = Excellent  ⚠️ = Limité  ❌ = Non disponible")

def show_next_steps():
    """
    Affiche les prochaines étapes
    """
    print("\n🚀 PROCHAINES ÉTAPES")
    print("-" * 30)
    
    steps = [
        "1. 🧪 Tester l'entraînement complet: python scripts/mnist_tensorboard_training.py",
        "2. 📊 Explorer les métriques dans TensorBoard",
        "3. 🔧 Personnaliser la configuration: scripts/config_tensorboard.py",
        "4. 📚 Consulter le guide complet: docs/TENSORBOARD_GUIDE.md",
        "5. 🎛️ Expérimenter avec différents hyperparamètres",
        "6. 📈 Comparer plusieurs architectures de modèles"
    ]
    
    for step in steps:
        print(step)

def main():
    """
    Fonction principale de démonstration
    """
    print_banner()
    
    # Vérifier l'installation
    if not check_installation():
        print("\n❌ Installation incomplète. Vérifiez les prérequis.")
        return False
    
    # Afficher la comparaison
    show_comparison()
    
    # Demander à l'utilisateur
    print("\n🎯 QUE VOULEZ-VOUS FAIRE ?")
    print("1. 🧪 Lancer un test rapide")
    print("2. 📊 Lancer TensorBoard (si des logs existent)")
    print("3. 🚀 Voir les prochaines étapes")
    print("4. ❌ Quitter")
    
    try:
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == "1":
            if run_quick_test():
                print("\n✅ Test réussi! Vous pouvez maintenant lancer TensorBoard.")
                if input("Lancer TensorBoard maintenant? (y/n): ").lower() == 'y':
                    launch_tensorboard_demo()
        
        elif choice == "2":
            launch_tensorboard_demo()
        
        elif choice == "3":
            show_next_steps()
        
        elif choice == "4":
            print("👋 Au revoir!")
            return True
        
        else:
            print("❌ Choix invalide")
            return False
    
    except KeyboardInterrupt:
        print("\n👋 Démonstration interrompue")
        return True
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 