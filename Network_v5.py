import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    
    def __init__(self, class_n):
        super(Network, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, class_n),
            nn.LeakyReLU()
        )
        self.feature_layer = nn.Sequential(
            nn.BatchNorm1d(9),
            nn.Linear(9, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, class_n),
            nn.LeakyReLU()
        )
        
    def forward(self, image, feature):
        output1 = self.model(image)
        output1 = self.fc_layer(output1)

        output2 = self.feature_layer(feature)

        output = (output1+output2)/2

        return output