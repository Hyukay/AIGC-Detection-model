import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model, loader):
    """
    Evaluates the model on a DataLoader.
    Args:
        model (nn.Module): Model to evaluate
        loader (DataLoader): DataLoader for validation/test set
    """
    model.eval().to(DEVICE)
    accuracies = []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        accuracies.append(accuracy)

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f"Validation/Test Accuracy: {avg_accuracy:.4f}")
