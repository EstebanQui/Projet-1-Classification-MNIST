import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import time

class MNISTNet(nn.Module):
    """
    Architecture CNN pour la classification MNIST avec TensorBoard
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # MaxPooling 2x2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Couches fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolution + ReLU + MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_with_tensorboard(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Entraînement avec TensorBoard pour le suivi des métriques
    """
    
    # Initialisation de TensorBoard
    log_dir = f"runs/mnist_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    # Composants d'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Variables de suivi
    best_accuracy = 0.0
    global_step = 0
    
    print(f"TensorBoard logs: {log_dir}")
    print("Démarrage de l'entraînement avec TensorBoard...")
    
    for epoch in range(num_epochs):
        print(f"\n=== ÉPOQUE {epoch+1}/{num_epochs} ===")
        
        # Mode entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistiques
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Log TensorBoard pour chaque batch
            if batch_idx % 50 == 0:  # Log tous les 50 batches
                writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
                batch_accuracy = 100. * predicted.eq(targets).sum().item() / targets.size(0)
                writer.add_scalar('Training/Batch_Accuracy', batch_accuracy, global_step)
                
                # Log des gradients (optionnel)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
            
            global_step += 1
        
        # Calcul des métriques d'époque pour l'entraînement
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_accuracy = 100. * train_correct / train_total
        
        # Évaluation sur le test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_accuracy = 100. * test_correct / test_total
        
        # Log TensorBoard pour les métriques d'époque
        writer.add_scalar('Epoch/Train_Loss', epoch_train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', epoch_train_accuracy, epoch)
        writer.add_scalar('Epoch/Test_Loss', epoch_test_loss, epoch)
        writer.add_scalar('Epoch/Test_Accuracy', epoch_test_accuracy, epoch)
        
        # Log de la différence train/test pour détecter l'overfitting
        writer.add_scalar('Epoch/Train_Test_Accuracy_Diff', 
                         abs(epoch_train_accuracy - epoch_test_accuracy), epoch)
        
        # Log des paramètres du modèle (une fois par époque)
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
        
        # Affichage des résultats
        print(f"Train - Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_accuracy:.2f}%")
        print(f"Test  - Loss: {epoch_test_loss:.4f} | Accuracy: {epoch_test_accuracy:.2f}%")
        
        # Sauvegarde du meilleur modèle
        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'loss': epoch_test_loss,
            }, 'models/mnist_model_best_tensorboard.pth')
            print(f"Nouveau meilleur modèle! Accuracy: {best_accuracy:.2f}%")
    
    # Log final des métriques
    writer.add_text('Training_Summary', 
                   f"Meilleure accuracy: {best_accuracy:.2f}%\n"
                   f"Époques: {num_epochs}\n"
                   f"Learning rate: {learning_rate}", 0)
    
    # Ajouter un graphique de comparaison train vs test
    writer.add_figure('Training_Curves', create_training_curves_figure(), 0)
    
    writer.close()
    print(f"\nEntraînement terminé! Logs TensorBoard: {log_dir}")
    print(f"Pour lancer TensorBoard: tensorboard --logdir={log_dir}")
    
    return model, best_accuracy

def create_training_curves_figure():
    """
    Crée une figure matplotlib pour TensorBoard
    """
    import matplotlib.pyplot as plt
    
    # Cette fonction serait appelée après l'entraînement
    # pour créer un graphique de comparaison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Exemple de courbes (à remplacer par les vraies données)
    epochs = range(1, 11)
    train_loss = [2.3, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1]
    test_loss = [2.4, 1.9, 1.3, 0.9, 0.6, 0.4, 0.3, 0.25, 0.22, 0.2]
    
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss')
    ax1.set_title('Évolution de la Loss')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    train_acc = [45, 65, 78, 85, 90, 93, 95, 96, 97, 98]
    test_acc = [44, 64, 77, 84, 89, 92, 94, 95, 96, 97]
    
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, test_acc, 'r-', label='Test Accuracy')
    ax2.set_title('Évolution de l\'Accuracy')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def launch_tensorboard(log_dir="runs"):
    """
    Lance TensorBoard pour visualiser les logs
    """
    import subprocess
    import webbrowser
    import time
    
    print(f"Lancement de TensorBoard depuis: {log_dir}")
    
    try:
        # Lancer TensorBoard en arrière-plan
        process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--port', '6006'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Attendre que TensorBoard démarre
        time.sleep(3)
        
        # Ouvrir le navigateur
        webbrowser.open('http://localhost:6006')
        
        print("TensorBoard lancé avec succès!")
        print("URL: http://localhost:6006")
        print("Appuyez sur Ctrl+C pour arrêter TensorBoard")
        
        return process
        
    except Exception as e:
        print(f"Erreur lors du lancement de TensorBoard: {e}")
        print("Assurez-vous que TensorBoard est installé: pip install tensorboard")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation
    print("Script d'intégration TensorBoard pour MNIST")
    print("=" * 50)
    
    # Vérifier l'installation de TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ TensorBoard disponible")
    except ImportError:
        print("✗ TensorBoard non installé")
        print("Installation: pip install tensorboard")
        exit(1)
    
    # Créer le dossier de logs
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Prêt pour l'intégration TensorBoard!")
    print("Utilisez train_with_tensorboard() pour l'entraînement")
    print("Utilisez launch_tensorboard() pour visualiser les résultats") 