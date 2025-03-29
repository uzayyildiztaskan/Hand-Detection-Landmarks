import torch.nn as nn
import torchvision.models as models

class HandLandmarkModel(nn.Module):
    def __init__(self):
        super(HandLandmarkModel, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.out_features = resnet.fc.in_features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.detection_head = nn.Sequential(
            nn.Linear(self.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
        
        self.keypoint_head = nn.Sequential(
            nn.Linear(self.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 42),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        bbox = self.detection_head(x)
        keypoints = self.keypoint_head(x)
        keypoints = keypoints.view(-1, 21, 2)
        return bbox, keypoints
