#!/usr/bin/env python3
"""
Script de lancement TensorBoard optimisé pour Windows
Résout les problèmes de PATH et de plugins dupliqués
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def find_tensorboard_executable():
    """
    Trouve l'exécutable TensorBoard sur Windows
    """
    possible_paths = [
        # Chemin standard après installation pip
        os.path.expanduser("~/AppData/Roaming/Python/Python313/Scripts/tensorboard.exe"),
        os.path.expanduser("~/AppData/Local/Programs/Python/Python313/Scripts/tensorboard.exe"),
        # Chemin système
        "C:/Python313/Scripts/tensorboard.exe",
        "C:/Python312/Scripts/tensorboard.exe",
        "C:/Python311/Scripts/tensorboard.exe",
        # Chemin conda
        os.path.expanduser("~/anaconda3/Scripts/tensorboard.exe"),
        os.path.expanduser("~/miniconda3/Scripts/tensorboard.exe"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ TensorBoard trouvé: {path}")
            return path
    
    # Essayer via Python
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import tensorboard; print(tensorboard.__file__)"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ TensorBoard disponible via Python")
            return "python"
    except:
        pass
    
    print("❌ TensorBoard non trouvé")
    return None

def launch_tensorboard_simple(log_dir="runs", port=6006):
    """
    Lance TensorBoard avec une approche simplifiée
    """
    print(f"🚀 Lancement de TensorBoard...")
    print(f"📁 Dossier de logs: {os.path.abspath(log_dir)}")
    print(f"🌐 Port: {port}")
    
    # Vérifier que le dossier de logs existe
    if not os.path.exists(log_dir):
        print(f"❌ Dossier de logs introuvable: {log_dir}")
        return False
    
    # Vérifier qu'il y a des logs
    log_files = list(Path(log_dir).glob("*"))
    if not log_files:
        print(f"❌ Aucun fichier de log trouvé dans {log_dir}")
        return False
    
    print(f"📊 {len(log_files)} expériences trouvées")
    
    # Trouver l'exécutable TensorBoard
    tb_executable = find_tensorboard_executable()
    if not tb_executable:
        print("❌ TensorBoard non trouvé. Installation: pip install tensorflow")
        return False
    
    # Préparer la commande
    if tb_executable == "python":
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--port", str(port),
            "--bind_all", "false"
        ]
    else:
        cmd = [
            tb_executable,
            "--logdir", log_dir,
            "--port", str(port),
            "--bind_all", "false"
        ]
    
    print(f"🚀 Commande: {' '.join(cmd)}")
    
    try:
        # Lancer TensorBoard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Attendre que le processus démarre
        time.sleep(3)
        
        # Vérifier si le processus fonctionne
        if process.poll() is None:
            print("✅ TensorBoard démarré avec succès!")
            
            # Attendre un peu plus pour que le serveur soit prêt
            time.sleep(2)
            
            # Ouvrir le navigateur
            url = f"http://localhost:{port}"
            print(f"🌐 Ouverture de {url}...")
            
            try:
                webbrowser.open(url)
                print("✅ Navigateur ouvert!")
            except Exception as e:
                print(f"⚠️ Impossible d'ouvrir le navigateur: {e}")
                print(f"Ouvrez manuellement: {url}")
            
            print("\n" + "="*50)
            print("📊 TENSORBOARD ACTIF")
            print("="*50)
            print(f"URL: {url}")
            print("Pour arrêter: Ctrl+C")
            print("="*50)
            
            # Attendre que l'utilisateur arrête
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
            print(f"❌ Erreur TensorBoard:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def launch_alternative_method(log_dir="runs", port=6006):
    """
    Méthode alternative utilisant un serveur HTTP simple
    """
    print("🔄 Tentative avec méthode alternative...")
    
    try:
        # Créer un serveur HTTP simple pour afficher les logs
        import http.server
        import socketserver
        
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=log_dir, **kwargs)
        
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            print(f"✅ Serveur HTTP démarré sur le port {port}")
            print(f"🌐 URL: http://localhost:{port}")
            
            # Ouvrir le navigateur
            webbrowser.open(f"http://localhost:{port}")
            
            print("📁 Affichage des fichiers de logs TensorBoard")
            print("🛑 Pour arrêter: Ctrl+C")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Arrêt du serveur...")
                httpd.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur serveur HTTP: {e}")
        return False

def main():
    """
    Fonction principale
    """
    print("🚀 LANCEUR TENSORBOARD WINDOWS")
    print("=" * 50)
    
    # Vérifier l'installation
    try:
        import tensorboard
        print("✅ TensorBoard installé")
    except ImportError:
        print("❌ TensorBoard non installé")
        print("Installation: pip install tensorflow")
        return False
    
    # Lister les expériences
    log_dir = "runs"
    if os.path.exists(log_dir):
        experiments = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if experiments:
            print(f"📁 Expériences trouvées: {len(experiments)}")
            for exp in experiments[:3]:  # Afficher les 3 premières
                print(f"   🔬 {exp}")
            if len(experiments) > 3:
                print(f"   ... et {len(experiments) - 3} autres")
        else:
            print("⚠️ Aucune expérience trouvée")
    else:
        print("⚠️ Dossier runs/ non trouvé")
    
    print("-" * 50)
    
    # Essayer le lancement principal
    if launch_tensorboard_simple(log_dir, 6006):
        return True
    
    # Si ça échoue, essayer la méthode alternative
    print("\n🔄 Tentative avec méthode alternative...")
    return launch_alternative_method(log_dir, 6006)

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 SOLUTIONS ALTERNATIVES:")
        print("1. Ouvrez un terminal en tant qu'administrateur")
        print("2. Essayez: pip install --upgrade tensorflow")
        print("3. Utilisez: python -m tensorboard.main --logdir=runs --port=6006")
        print("4. Ou lancez la démonstration: python scripts/demo_tensorboard.py")
    
    input("\nAppuyez sur Entrée pour quitter...") 