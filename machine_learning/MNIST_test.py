import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Callable, Optional


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from ImplicitConvParam import ImplicitParameterizerConvolutional


class MNISTClassifier(nn.Module):
    def __init__(self, n=32, solver_params=None):
        super().__init__()

        if solver_params is None:
            solver_params = {
                "SOLVER_ALPHA": 0.5,
                "SOLVER_MAX_ITER": 50,
                "SOLVER_TOL": 1e-4
            }

        # Use the convolutional implicit parameterizer
        # MNIST images are 28x28, we'll resize to 32x32 for convenience
        self.implicit_conv = ImplicitParameterizerConvolutional(
            input_dim=1,  # Grayscale MNIST
            output_dim=10,  # 10 classes
            n=n,
            hidden_dim=32,
            activation=torch.relu,
            solver_params=solver_params
        )

    def forward(self, x):
        # Resize from 28x28 to 32x32
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        return self.implicit_conv(x)


def train_mnist():
    # Set random seed
    torch.manual_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Create model
    print("Creating model...")
    solver_params = {
        "SOLVER_ALPHA": 0.5,
        "SOLVER_MAX_ITER": 30,
        "SOLVER_TOL": 1e-4
    }
    model = MNISTClassifier(n=32, solver_params=solver_params).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    n_epochs = 20
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print("\nTraining...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # if (batch_idx + 1) % 100 == 0:
            #     print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{n_epochs}| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% |  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(test_accs, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('mnist_training.png')
    print("Training curves saved to mnist_training.png")

    # Visualize some predictions
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)

        # Make predictions
        outputs = model(images)
        _, predicted = outputs.max(1)

        # Move to CPU for plotting
        images = images.cpu()
        labels = labels.cpu()
        predicted = predicted.cpu()

        # Plot first 16 images
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                img = images[idx].squeeze()
                ax.imshow(img, cmap='gray')
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('mnist_predictions.png')
        print("Sample predictions saved to mnist_predictions.png")

    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")

    return model


class CIFAR10Classifier(nn.Module):
    def __init__(self, n=64, solver_params=None): # Increased n for color images
        super().__init__()
        
        if solver_params is None:
            solver_params = {
                "SOLVER_ALPHA": 0.5,
                "SOLVER_MAX_ITER": 50,
                "SOLVER_TOL": 1e-4
            }
        
        # CIFAR10 images are 32x32x3
        self.implicit_conv = ImplicitParameterizerConvolutional(
            input_dim=3,      # Change: 3 channels for RGB
            output_dim=10,    # 10 CIFAR classes
            n=n,
            hidden_dim=64,   # Increased hidden_dim for higher complexity
            activation=torch.relu,
            solver_params=solver_params
        )
    
    def forward(self, x):
        # Change: Removed F.interpolate as CIFAR is already 32x32
        return self.implicit_conv(x)

def train_cifar10():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Change: CIFAR10 specific normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Added augmentation for CIFAR
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Change: Using n=64 for better capacity on CIFAR
    model = CIFAR10Classifier(n=32).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    
    # Training setup - Reduced LR slightly for stability on CIFAR
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.95, weight_decay=5e-4)

    n_epochs = 200 # CIFAR usually takes longer to converge than MNIST


    total_steps = len(train_loader) * n_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-2,          # Peak LR
        total_steps=total_steps,
        pct_start=30/65,      # Rise for 30 epochs
        div_factor=10,        # Start at 1e-3
        final_div_factor=1,   # End at 1e-3
        anneal_strategy='linear'
    )    

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
    for epoch in range(n_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Critical: Keep the gradient clipping from our stability fix
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() # Note: OneCycleLR is usually called per-batch
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Stats and Evaluation (same logic as MNIST)
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        model.eval()
        test_l, test_c, test_t = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_l += criterion(out, labels).item()
                _, pred = out.max(1)
                test_t += labels.size(0)
                test_c += pred.eq(labels).sum().item()
        
        test_acc = 100. * test_c / test_t
        train_losses.append(train_loss); train_accs.append(train_acc)
        test_losses.append(test_l/len(test_loader)); test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

    return model

