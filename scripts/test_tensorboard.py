#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration TensorBoard
Génère des données de démonstration pour tester les visualisations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import time

def create_dummy_model():
    """
    Crée un modèle factice pour les tests
    """
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    return DummyModel()

def generate_dummy_data():
    """
    Génère des données factices pour les tests
    """
    # Données d'entraînement factices
    train_data = torch.randn(1000, 1, 28, 28)
    train_labels = torch.randint(0, 10, (1000,))
    
    # Données de test factices
    test_data = torch.randn(200, 1, 28, 28)
    test_labels = torch.randint(0, 10, (200,))
    
    return train_data, train_labels, test_data, test_labels

def test_tensorboard_integration():
    """
    Test complet de l'intégration TensorBoard
    """
    print("🧪 TEST D'INTÉGRATION TENSORBOARD")
    print("=" * 50)
    
    # Vérifier l'installation de TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'import TensorBoard: {e}")
        print("Installation: pip install tensorboard")
        return False
    
    # Créer le dossier de logs
    log_dir = f"runs/test_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialiser TensorBoard
    try:
        writer = SummaryWriter(log_dir)
        print(f"✅ Writer TensorBoard créé: {log_dir}")
    except Exception as e:
        print(f"❌ Erreur création writer: {e}")
        return False
    
    # Créer le modèle et les données
    model = create_dummy_model()
    train_data, train_labels, test_data, test_labels = generate_dummy_data()
    
    # Composants d'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🚀 Démarrage du test d'entraînement factice...")
    
    # Simulation d'entraînement
    num_epochs = 5
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"📊 Époque {epoch+1}/{num_epochs}")
        
        # Mode entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Simulation de batches
        num_batches = 10
        for batch in range(num_batches):
            # Données factices
            batch_data = train_data[batch*100:(batch+1)*100]
            batch_labels = train_labels[batch*100:(batch+1)*100]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistiques
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
            
            # Log TensorBoard
            if batch % 2 == 0:  # Log tous les 2 batches
                writer.add_scalar('Test/Batch_Loss', loss.item(), global_step)
                batch_accuracy = 100. * predicted.eq(batch_labels).sum().item() / batch_labels.size(0)
                writer.add_scalar('Test/Batch_Accuracy', batch_accuracy, global_step)
                
                # Log des gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Test/Gradients/{name}', param.grad, global_step)
            
            global_step += 1
        
        # Métriques d'époque
        epoch_train_loss = train_loss / num_batches
        epoch_train_accuracy = 100. * train_correct / train_total
        
        # Évaluation factice
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            test_loss = criterion(test_outputs, test_labels)
            _, test_predicted = test_outputs.max(1)
            test_accuracy = 100. * test_predicted.eq(test_labels).sum().item() / test_labels.size(0)
        
        # Log des métriques d'époque
        writer.add_scalar('Test/Epoch_Train_Loss', epoch_train_loss, epoch)
        writer.add_scalar('Test/Epoch_Train_Accuracy', epoch_train_accuracy, epoch)
        writer.add_scalar('Test/Epoch_Test_Loss', test_loss.item(), epoch)
        writer.add_scalar('Test/Epoch_Test_Accuracy', test_accuracy, epoch)
        writer.add_scalar('Test/Train_Test_Diff', abs(epoch_train_accuracy - test_accuracy), epoch)
        
        # Log des paramètres
        for name, param in model.named_parameters():
            writer.add_histogram(f'Test/Parameters/{name}', param.data, epoch)
        
        print(f"   Train - Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_accuracy:.2f}%")
        print(f"   Test  - Loss: {test_loss.item():.4f} | Acc: {test_accuracy:.2f}%")
    
    # Log final
    writer.add_text('Test/Summary', 
                   f"Test d'intégration TensorBoard terminé\n"
                   f"Époques: {num_epochs}\n"
                   f"Logs générés: {log_dir}\n"
                   f"Timestamp: {datetime.now().isoformat()}", 0)
    
    # Ajouter quelques images factices
    try:
        dummy_images = torch.randn(16, 1, 28, 28)  # 16 images factices
        writer.add_images('Test/Sample_Images', dummy_images, 0)
        print("✅ Images factices ajoutées")
    except Exception as e:
        print(f"⚠️ Erreur ajout images: {e}")
    
    writer.close()
    
    print(f"✅ Test terminé avec succès!")
    print(f"📁 Logs générés: {log_dir}")
    print(f"📊 Métriques loggées:")
    print(f"   - Loss et Accuracy par batch")
    print(f"   - Métriques par époque")
    print(f"   - Histogrammes des paramètres")
    print(f"   - Distribution des gradients")
    print(f"   - Images d'exemple")
    
    return True, log_dir

def verify_log_files(log_dir):
    """
    Vérifie que les fichiers de log ont été créés
    """
    print(f"\n🔍 VÉRIFICATION DES FICHIERS DE LOG")
    print("-" * 30)
    
    if not os.path.exists(log_dir):
        print(f"❌ Dossier de logs introuvable: {log_dir}")
        return False
    
    # Lister les fichiers
    files = os.listdir(log_dir)
    event_files = [f for f in files if f.startswith('events.out.tfevents')]
    
    print(f"📁 Dossier: {log_dir}")
    print(f"📄 Fichiers trouvés: {len(files)}")
    print(f"📊 Fichiers d'événements: {len(event_files)}")
    
    if event_files:
        print("✅ Fichiers TensorBoard créés avec succès")
        for file in event_files:
            file_path = os.path.join(log_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   📄 {file} ({file_size} bytes)")
        return True
    else:
        print("❌ Aucun fichier d'événement trouvé")
        return False

def main():
    """
    Fonction principale du test
    """
    print("🧪 DÉMARRAGE DU TEST TENSORBOARD")
    print("=" * 50)
    
    # Créer le dossier runs s'il n'existe pas
    os.makedirs("runs", exist_ok=True)
    
    # Exécuter le test
    success, log_dir = test_tensorboard_integration()
    
    if success:
        # Vérifier les fichiers
        verify_log_files(log_dir)
        
        print("\n" + "="*50)
        print("🎉 TEST RÉUSSI!")
        print("="*50)
        print("Pour visualiser les résultats:")
        print(f"1. Lancez: tensorboard --logdir=runs")
        print(f"2. Ouvrez: http://localhost:6006")
        print(f"3. Cherchez l'expérience: {os.path.basename(log_dir)}")
        
        return True
    else:
        print("\n" + "="*50)
        print("❌ TEST ÉCHOUÉ!")
        print("="*50)
        print("Vérifiez:")
        print("1. Installation de TensorBoard: pip install tensorboard")
        print("2. Permissions d'écriture dans le dossier runs/")
        print("3. Espace disque disponible")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 