"""
Script d'entrainement MNIST sans torchvision
Compatible avec Python 3.13.5 et PyTorch 2.7.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import requests
import gzip
import os
from datetime import datetime

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilise: {device}")

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def download_mnist():
    """Telecharge et charge les donnees MNIST manuellement"""
    urls = {
        'train-images-idx3-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    }

    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    def load_images(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            url = urls[filename]
            print(f"Telechargement: {filename}")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
            magic = int.from_bytes(data[0:4], 'big')
            num_images = int.from_bytes(data[4:8], 'big')
            rows = int.from_bytes(data[8:12], 'big')
            cols = int.from_bytes(data[12:16], 'big')
            
            images = np.frombuffer(data[16:], dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
            return images
    
    def load_labels(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            url = urls[filename]
            print(f"Telechargement: {filename}")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
            magic = int.from_bytes(data[0:4], 'big')
            num_labels = int.from_bytes(data[4:8], 'big')
            labels = np.frombuffer(data[8:], dtype=np.uint8)
            return labels
    
    print("Chargement des donnees MNIST...")
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')
    
    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels):
    """Preprocessing des donnees MNIST"""
    images = images.astype(np.float32) / 255.0
    
    # Normalisation standard MNIST
    mean = 0.1307
    std = 0.3081
    images = (images - mean) / std
    
    # Ajouter dimension channel
    images = images[:, np.newaxis, :, :]
    
    images_tensor = torch.from_numpy(images)
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    
    return images_tensor, labels_tensor

def create_data_loaders(batch_size=64):
    """Cree les DataLoaders"""
    train_images, train_labels, test_images, test_labels = download_mnist()
    
    print("Preprocessing des donnees...")
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraine le modele pour une epoque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total

def test_model(model, test_loader, criterion, device):
    """Evalue le modele sur le dataset de test"""
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
    
    return test_loss / len(test_loader), 100. * correct / total

def save_model(model, path):
    """Sauvegarde le modele"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
        'timestamp': datetime.now().isoformat()
    }, path)
    print(f"Modele sauvegarde: {path}")

def main():
    """Fonction principale"""
    print("=== Entrainement MNIST ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    
    # Hyperparametres
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 3  # Reduit pour test rapide
    
    # Chargement des donnees
    train_loader, test_loader = create_data_loaders(batch_size)
    print(f"Dataset d'entrainement: {len(train_loader.dataset)} echantillons")
    print(f"Dataset de test: {len(test_loader.dataset)} echantillons")
    
    # Creation du modele
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModele cree avec {sum(p.numel() for p in model.parameters())} parametres")
    
    # Entrainement
    print(f"\nDebut de l'entrainement ({num_epochs} epoques)...")
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoque {epoch+1}/{num_epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_model(model, '../models/mnist_model_best.pth')
    
    save_model(model, '../models/mnist_model_final.pth')
    
    print(f"\n=== Entrainement termine ===")
    print(f"Meilleure precision: {best_accuracy:.2f}%")
    print("Modeles sauvegardes dans ../models/")

if __name__ == "__main__":
    main() 