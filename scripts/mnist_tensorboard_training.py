import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import requests
import gzip
import os
from datetime import datetime
import time

# Configuration PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print("Configuration du système")
print(f"Device utilisé: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU non disponible, utilisation du CPU")

print("\n" + "="*50)

class MNISTNet(nn.Module):
    """
    Architecture CNN pour la classification MNIST
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def download_mnist():
    """
    Télécharge et charge le dataset MNIST
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)
    
    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    # Télécharger et charger les données
    train_images = load_mnist_images(os.path.join(data_dir, files['train_images']))
    train_labels = load_mnist_labels(os.path.join(data_dir, files['train_labels']))
    test_images = load_mnist_images(os.path.join(data_dir, files['test_images']))
    test_labels = load_mnist_labels(os.path.join(data_dir, files['test_labels']))
    
    print(f"Dataset MNIST chargé:")
    print(f"Train: {train_images.shape[0]} images, {train_labels.shape[0]} labels")
    print(f"Test: {test_images.shape[0]} images, {test_labels.shape[0]} labels")
    
    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels):
    """
    Prétraitement des données MNIST
    """
    # Normalisation [0,255] → [0,1]
    images = images.astype(np.float32) / 255.0
    
    # Standardisation MNIST
    mean = 0.1307
    std = 0.3081
    images = (images - mean) / std
    
    # Ajouter la dimension canal
    images = images[:, np.newaxis, :, :]
    
    # Conversion en tenseurs PyTorch
    images_tensor = torch.from_numpy(images)
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    
    return images_tensor, labels_tensor

def create_data_loaders(batch_size=128):
    """
    Crée les DataLoaders PyTorch
    """
    print("Prétraitement des données...")
    
    # Charger les données
    train_images, train_labels, test_images, test_labels = download_mnist()
    
    # Prétraitement
    train_images_tensor, train_labels_tensor = preprocess_data(train_images, train_labels)
    test_images_tensor, test_labels_tensor = preprocess_data(test_images, test_labels)
    
    # Créer les datasets PyTorch
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    
    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"DataLoaders créés:")
    print(f"   Training: {len(train_loader)} batches de {batch_size} images")
    print(f"   Test: {len(test_loader)} batches de {batch_size} images")
    
    return train_loader, test_loader

def train_epoch_with_tensorboard(model, train_loader, criterion, optimizer, device, epoch_num, writer, global_step):
    """
    Entraîne le modèle pour une époque avec TensorBoard
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log TensorBoard pour chaque batch
        if batch_idx % 50 == 0:
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            batch_accuracy = 100. * predicted.eq(targets).sum().item() / targets.size(0)
            writer.add_scalar('Training/Batch_Accuracy', batch_accuracy, global_step)
            
            # Log des gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
        
        global_step += 1
    
    # Résultats finaux de l'époque
    final_loss = running_loss / len(train_loader)
    final_accuracy = 100. * correct / total
    
    print(f'Époque {epoch_num} terminée - Loss: {final_loss:.4f} | Accuracy: {final_accuracy:.2f}%')
    
    return final_loss, final_accuracy, global_step

def evaluate_model_with_tensorboard(model, test_loader, criterion, device, epoch, writer):
    """
    Évalue le modèle avec TensorBoard
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    final_loss = test_loss / len(test_loader)
    final_accuracy = 100. * correct / total
    
    return final_loss, final_accuracy

def save_model(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Sauvegarde le modèle
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'model_architecture': 'MNISTNet CNN'
    }, filepath)
    
    print(f"Modèle sauvegardé: {filepath}")

def main():
    """
    Fonction principale avec TensorBoard
    """
    # Hyperparamètres
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    
    print("Début de l'entraînement MNIST avec TensorBoard")
    print(f"Configuration:")
    print(f"Époques: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Initialisation de TensorBoard
    log_dir = f"runs/mnist_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    # Créer les DataLoaders
    train_loader, test_loader = create_data_loaders(BATCH_SIZE)
    
    # Initialisation du modèle et des composants d'entraînement
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Variables pour suivre le meilleur modèle
    best_accuracy = 0.0
    global_step = 0
    
    print(f"TensorBoard logs: {log_dir}")
    
    # Boucle d'entraînement principale
    total_start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== ÉPOQUE {epoch}/{NUM_EPOCHS} ===")
        
        # Entraînement sur une époque
        train_loss, train_acc, global_step = train_epoch_with_tensorboard(
            model, train_loader, criterion, optimizer, device, epoch, writer, global_step
        )
        
        # Évaluation sur le dataset de test
        print("Évaluation sur le dataset de test...")
        test_loss, test_acc = evaluate_model_with_tensorboard(
            model, test_loader, criterion, device, epoch, writer
        )
        
        # Log TensorBoard pour les métriques d'époque
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Test_Loss', test_loss, epoch)
        writer.add_scalar('Epoch/Test_Accuracy', test_acc, epoch)
        writer.add_scalar('Epoch/Train_Test_Accuracy_Diff', 
                         abs(train_acc - test_acc), epoch)
        
        # Log des paramètres du modèle
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
        
        # Affichage des résultats
        print(f"Résultats Époque {epoch}:")
        print(f"Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"Test  - Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_model(model, optimizer, epoch, test_loss, test_acc, 'models/mnist_model_best_tensorboard.pth')
            print(f"Nouveau meilleur modèle! Accuracy: {best_accuracy:.2f}%")
        
        print("=" * 60)
    
    # Statistiques finales
    total_time = time.time() - total_start_time
    
    # Log final des métriques
    writer.add_text('Training_Summary', 
                   f"Meilleure accuracy: {best_accuracy:.2f}%\n"
                   f"Époques: {NUM_EPOCHS}\n"
                   f"Learning rate: {LEARNING_RATE}\n"
                   f"Temps total: {total_time:.1f}s", 0)
    
    writer.close()
    
    print(f"ENTRAÎNEMENT TERMINÉ!")
    print(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Meilleure accuracy: {best_accuracy:.2f}%")
    print(f"Logs TensorBoard: {log_dir}")
    print(f"Pour lancer TensorBoard: tensorboard --logdir={log_dir}")
    
    # Sauvegarder le modèle final
    save_model(model, optimizer, NUM_EPOCHS, test_loss, test_acc, 'models/mnist_model_final_tensorboard.pth')
    
    return model, best_accuracy

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Lancer l'entraînement
    model, best_accuracy = main()
    
    print("\n" + "="*50)
    print("ENTRAÎNEMENT AVEC TENSORBOARD TERMINÉ!")
    print("="*50)
    print("Pour visualiser les résultats:")
    print("1. Ouvrez un terminal")
    print("2. Naviguez vers ce dossier")
    print("3. Lancez: tensorboard --logdir=runs")
    print("4. Ouvrez http://localhost:6006 dans votre navigateur") 