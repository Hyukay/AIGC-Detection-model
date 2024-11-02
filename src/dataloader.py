from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import RealFakeDataset

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates train, validation, and test dataloaders with specified transformations.
    Args:
        data_dir (str): Root directory containing the dataset
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of worker processes for data loading
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transformations specific to ConvNeXt input size and requirements
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),  # ConvNeXt standard size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RealFakeDataset(data_dir, transform=TRAIN_TRANSFORM, split="train")
    val_dataset = RealFakeDataset(data_dir, transform=VAL_TRANSFORM, split="val")
    test_dataset = RealFakeDataset(data_dir, transform=VAL_TRANSFORM, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
