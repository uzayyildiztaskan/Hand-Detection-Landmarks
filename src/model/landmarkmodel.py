import torch.nn as nn
import torchvision.models as models

class HandLandmarkModel(nn.Module):
    def __init__(self):
        super(HandLandmarkModel, self).__init__()
        
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        self.detector = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4)
        )
        
        self.keypointer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 42)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        bbox = self.detector(features)
        keypoints = self.keypointer(features).view(-1, 21, 2)
        return bbox, keypoints
