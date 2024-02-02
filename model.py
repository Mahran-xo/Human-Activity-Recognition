# from torchvision.models import EfficientNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class EffNet(nn.Module):
    def __init__(self, num_classes=15, dropout_prob=0.5):
        super(EffNet, self).__init__()
        eff = EfficientNet.from_pretrained('efficientnet-b0')  # or choose another variant
        self.features = eff.extract_features

        # Adding dropout to the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(eff._fc.in_features, num_classes)  # Assuming the FC layer is named 'fc'
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
