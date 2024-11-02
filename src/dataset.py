import os
from PIL import Image
from torch.utils.data import Dataset

class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        """
        Args:
            root_dir (str): Path to the `data/` directory
            transform (callable, optional): Optional transform to be applied on images
            split (str): 'train', 'val', or 'test' split
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data = []

        for label in ["0_real", "1_fake"]:
            class_dir = os.path.join(self.root_dir, label)
            if os.path.exists(class_dir):
                class_files = [
                    (os.path.join(class_dir, f), int(label[0]))
                    for f in os.listdir(class_dir)
                    if f.endswith((".png", ".jpg", ".jpeg"))
                ]
                self.data.extend(class_files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
