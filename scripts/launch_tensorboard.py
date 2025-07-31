#!/usr/bin/env python3
"""
Script pour lancer TensorBoard et visualiser les logs d'entraînement MNIST
"""

import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path

def launch_tensorboard(log_dir="runs", port=6006):
    """
    Lance TensorBoard et ouvre le navigateur
    
    Args:
        log_dir: Dossier contenant les logs TensorBoard
        port: Port sur lequel lancer TensorBoard
    """
    
    # Vérifier que le dossier de logs existe
    if not os.path.exists(log_dir):
        print(f"❌ Dossier de logs '{log_dir}' introuvable!")
        print("Assurez-vous d'avoir exécuté l'entraînement avec TensorBoard d'abord.")
        return False
    
    # Vérifier qu'il y a des logs
    log_files = list(Path(log_dir).glob("*"))
    if not log_files:
        print(f"❌ Aucun fichier de log trouvé dans '{log_dir}'")
        print("Assurez-vous d'avoir exécuté l'entraînement avec TensorBoard d'abord.")
        return False
    
    print(f"📊 Lancement de TensorBoard...")
    print(f"📁 Dossier de logs: {os.path.abspath(log_dir)}")
    print(f"🌐 Port: {port}")
    
    try:
        # Lancer TensorBoard en arrière-plan
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
        
        print(f"🚀 Commande: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Attendre que TensorBoard démarre
        print("⏳ Attente du démarrage de TensorBoard...")
        time.sleep(5)
        
        # Vérifier si le processus fonctionne
        if process.poll() is None:
            print("✅ TensorBoard démarré avec succès!")
            
            # Ouvrir le navigateur
            url = f"http://localhost:{port}"
            print(f"🌐 Ouverture de {url} dans le navigateur...")
            
            try:
                webbrowser.open(url)
                print("✅ Navigateur ouvert!")
            except Exception as e:
                print(f"⚠️ Impossible d'ouvrir le navigateur automatiquement: {e}")
                print(f"Ouvrez manuellement: {url}")
            
            print("\n" + "="*50)
            print("📊 TENSORBOARD ACTIF")
            print("="*50)
            print(f"URL: {url}")
            print("Pour arrêter TensorBoard: Ctrl+C")
            print("="*50)
            
            try:
                # Attendre que l'utilisateur arrête le processus
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Arrêt de TensorBoard...")
                process.terminate()
                process.wait()
                print("✅ TensorBoard arrêté.")
            
            return True
            
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Erreur lors du lancement de TensorBoard:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except FileNotFoundError:
        print("❌ TensorBoard non trouvé!")
        print("Installation: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def check_tensorboard_installation():
    """
    Vérifie si TensorBoard est installé
    """
    try:
        import tensorboard
        print("✅ TensorBoard est installé")
        return True
    except ImportError:
        print("❌ TensorBoard n'est pas installé")
        print("Installation: pip install tensorboard")
        return False

def list_experiments(log_dir="runs"):
    """
    Liste les expériences disponibles
    """
    if not os.path.exists(log_dir):
        print(f"Aucun dossier de logs trouvé: {log_dir}")
        return
    
    print(f"📁 Expériences disponibles dans '{log_dir}':")
    print("-" * 50)
    
    experiments = []
    for item in os.listdir(log_dir):
        item_path = os.path.join(log_dir, item)
        if os.path.isdir(item_path):
            # Compter les fichiers d'événements
            event_files = list(Path(item_path).glob("events.out.tfevents.*"))
            experiments.append((item, len(event_files)))
    
    if not experiments:
        print("Aucune expérience trouvée")
        return
    
    for name, event_count in sorted(experiments):
        print(f"🔬 {name} ({event_count} fichiers d'événements)")
    
    print("-" * 50)

if __name__ == "__main__":
    print("🚀 LANCEUR TENSORBOARD POUR MNIST")
    print("=" * 50)
    
    # Vérifier l'installation
    if not check_tensorboard_installation():
        sys.exit(1)
    
    # Lister les expériences disponibles
    list_experiments()
    
    # Lancer TensorBoard
    success = launch_tensorboard()
    
    if not success:
        print("\n💡 Conseils de dépannage:")
        print("1. Vérifiez que TensorBoard est installé: pip install tensorboard")
        print("2. Assurez-vous d'avoir exécuté l'entraînement avec TensorBoard")
        print("3. Vérifiez que le dossier 'runs' contient des logs")
        print("4. Essayez un port différent si le port 6006 est occupé")
        sys.exit(1) 