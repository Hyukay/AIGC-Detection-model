import torch
from tqdm import tqdm
from src.evaluate import evaluate
from utils.plot import plot_training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, optimizer, epochs=40, loss_fn=torch.nn.CrossEntropyLoss()):
    """
    Training loop for the model, tracks loss and accuracy.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training DataLoader
        val_loader (DataLoader): Validation DataLoader
        optimizer (torch.optim.Optimizer): Optimizer for training
        epochs (int): Number of epochs to train
        loss_fn (nn.Module): Loss function (default: CrossEntropyLoss)
    """
    model.to(DEVICE)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            preds = outputs.argmax(dim=1)
            epoch_train_correct += (preds == labels).sum().item()
            epoch_train_total += labels.size(0)
            epoch_train_loss += loss.item() * labels.size(0)

            # Update loop description
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), accuracy=(epoch_train_correct / epoch_train_total))

        # Calculate average training loss and accuracy for the epoch
        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_accuracy = epoch_train_correct / epoch_train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation phase
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Plot training and validation metrics
    plot_training(train_losses, val_losses, train_accuracies, val_accuracies)
