from models.BasicModule import BasicModule
from torch import nn


class AlexNet(BasicModule):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.num_classes_ = num_classes
        self.feature_layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3,  padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes_))

    def forward(self, input):

        net = self.feature_layers(input)
        net = net.view(-1, 256*6*6)
        return self.fc_layers(net)
