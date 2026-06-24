import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import pickle as pkl

# Custom Dataset to load images from directories
class DogDataset(Dataset):
    def __init__(self, dataset_root, classes, transform=None):
        self.dataset_root = dataset_root
        self.classes = classes
        self.transform = transform
        self.samples = []
        
        # Build samples list (filepath, class_id)
        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(dataset_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # Verify file extension and if it's a valid file
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append((img_path, class_id))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images
            image = Image.new('RGB', (299, 299), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, class_id

def main(args):
    # Parameters
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
    args.classes = os.path.expanduser(args.classes)
    
    # Load class names
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)
    print(f"Loading {num_classes} classes: {classes}")

    # Set up GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocessing Transforms
    # Xception expects 299x299. We also apply basic normalization matching ImageNet
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalization matching Xception (-1 to 1)
    ])

    # Load dataset
    full_dataset = DogDataset(args.dataset_root, classes, transform=transform)
    dataset_size = len(full_dataset)
    print(f"Total dataset size: {dataset_size} images")
    
    if dataset_size == 0:
        raise ValueError(f"No images found in dataset root: {args.dataset_root}")

    # Split dataset (Training / Validation)
    train_size = int(args.split * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training on {train_size} images, Validation on {val_size} images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_pre, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_pre, shuffle=False, num_workers=2)

    # ====================================================
    # Build Model (Transfer Learning)
    # ====================================================
    # In PyTorch, we can load a pre-trained model. We use ResNet50 or Inception_V3 as standard alternatives,
    # or timm.create_model('xception', pretrained=True) if the user has 'timm' installed.
    # Here, we will use torchvision's models.resnet50 or models.inception_v3. 
    # Let's use InceptionV3 since it shares the same input size (299x299) and depth characteristics as Xception.
    print("Initializing model...")
    model = models.inception_v3(pretrained=True, aux_logits=True)
    
    # Replace the classification head
    # InceptionV3 has model.fc (fully connected) for classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # ====================================================
    # Phase 1: Train only the top classifier (freeze backbone)
    # ====================================================
    print("\n--- Phase 1: Training classifier head (backbone frozen) ---")
    # Freeze all layers except model.fc
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    # Optimize only parameters of the head
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr_pre)

    os.makedirs(args.result_root, exist_ok=True)
    
    pre_train_accs, pre_val_accs = [], []
    pre_train_losses, pre_val_losses = [], []

    for epoch in range(args.epochs_pre):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Inception V3 returns a named tuple (logits, aux_logits) during training
            outputs, aux_outputs = model(inputs)
            loss = criterion(outputs, targets) + 0.4 * criterion(aux_outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        pre_train_accs.append(epoch_acc)
        pre_val_accs.append(epoch_val_acc)
        pre_train_losses.append(epoch_loss)
        pre_val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs_pre} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # Save intermediate model
    torch.save(model.state_dict(), os.path.join(args.result_root, 'model_pre_final.pt'))

    # ====================================================
    # Phase 2: Fine-tune the whole model
    # ====================================================
    print("\n--- Phase 2: Fine-tuning all layers ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    # Switch dataloaders to fine-tuning batch size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_fine, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_fine, shuffle=False, num_workers=2)

    # Lower learning rate for fine tuning the entire network
    optimizer = optim.Adam(model.parameters(), lr=args.lr_fine)

    fine_train_accs, fine_val_accs = [], []
    fine_train_losses, fine_val_losses = [], []

    for epoch in range(args.epochs_fine):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs, aux_outputs = model(inputs)
            loss = criterion(outputs, targets) + 0.4 * criterion(aux_outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        fine_train_accs.append(epoch_acc)
        fine_val_accs.append(epoch_val_acc)
        fine_train_losses.append(epoch_loss)
        fine_val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs_fine} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.result_root, 'model_fine_final.pt'))
    print(f"\nModel training finished. Saved to: {args.result_root}")

    # ====================================================
    # Save results and plot graphs
    # ====================================================
    acc = pre_train_accs + fine_train_accs
    val_acc = pre_val_accs + fine_val_accs
    loss = pre_train_losses + fine_train_losses
    val_loss = pre_val_losses + fine_val_losses
    total_epochs = args.epochs_pre + args.epochs_fine

    # Save graphs
    plt.figure()
    plt.plot(range(total_epochs), acc, marker='.', label='train_acc')
    plt.plot(range(total_epochs), val_acc, marker='.', label='val_acc')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(args.result_root, 'acc_pytorch.png'))
    plt.close()

    plt.figure()
    plt.plot(range(total_epochs), loss, marker='.', label='train_loss')
    plt.plot(range(total_epochs), val_loss, marker='.', label='val_loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.result_root, 'loss_pytorch.png'))
    plt.close()

    # Save plot data as pickle file
    plot_data = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }
    with open(os.path.join(args.result_root, 'plot_pytorch.dump'), 'wb') as f:
        pkl.dump(plot_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str, help='Root path to the dataset folder')
    parser.add_argument('classes', type=str, help='Path to the classes txt file')
    parser.add_argument('result_root', type=str, help='Folder where results will be saved')
    parser.add_argument('--epochs_pre', type=int, default=5, help='Epochs to train only the classifier head')
    parser.add_argument('--epochs_fine', type=int, default=50, help='Epochs to train the entire model')
    parser.add_argument('--batch_size_pre', type=int, default=32, help='Batch size for Phase 1')
    parser.add_argument('--batch_size_fine', type=int, default=16, help='Batch size for Phase 2')
    parser.add_argument('--lr_pre', type=float, default=1e-3, help='Learning rate for Phase 1')
    parser.add_argument('--lr_fine', type=float, default=1e-4, help='Learning rate for Phase 2')
    parser.add_argument('--split', type=float, default=0.8, help='Training/Validation split ratio')
    
    args = parser.parse_args()
    main(args)
