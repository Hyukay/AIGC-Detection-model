import torch
from src.dataloader import get_dataloaders
from src.model import ConvNext
from src.train import train
from utils.config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, EPOCHS, LEARNING_RATE, WEIGHT_DECAY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = ConvNext()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, epochs=EPOCHS, loss_fn=loss_fn)
    
    from src.evaluate import evaluate
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
