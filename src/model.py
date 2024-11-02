import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class ConvNext(nn.Module):
    def __init__(self, num_classes=2):
        """
        Initializes ConvNeXt with a custom classifier for binary classification.
        Args:
            num_classes (int): Number of output classes (default is 2 for real/fake).
        """
        super(ConvNext, self).__init__()
        # Load ConvNeXt base with pre-trained weights
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Freeze base layers
        for param in self.convnext.parameters():
            param.requires_grad = False

        # Replace classifier with custom head
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.convnext(x)
