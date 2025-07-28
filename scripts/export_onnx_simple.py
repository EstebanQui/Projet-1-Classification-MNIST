"""
Script d'export ONNX simplifié pour le modèle MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Redéfinition du modèle (évite les problèmes d'import)
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

def main():
    print("=== Export ONNX simplifié ===")
    
    # Chemins des fichiers
    model_path = '../models/mnist_model_best.pth'
    onnx_path = '../models/mnist_model.onnx'
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    print(f"📦 Chargement du modèle: {model_path}")
    
    # Charger le modèle
    try:
        model = MNISTNet()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # Créer un exemple d'entrée
    example_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    print("📤 Export vers ONNX...")
    
    # Export ONNX
    try:
        torch.onnx.export(
            model,                      # Modèle PyTorch
            example_input,              # Exemple d'entrée
            onnx_path,                  # Chemin de sortie
            export_params=True,         # Exporter les paramètres
            opset_version=11,           # Version ONNX
            do_constant_folding=True,   # Optimisation
            input_names=['input'],      # Noms des entrées
            output_names=['output'],    # Noms des sorties
            dynamic_axes={              # Dimensions dynamiques
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Export ONNX réussi: {onnx_path}")
        
        # Vérifier la taille
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"📊 Taille du modèle ONNX: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export ONNX: {e}")
        return
    
    print("🎉 Export terminé avec succès!")

if __name__ == "__main__":
    main() 