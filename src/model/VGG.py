from src.common_imports import *
from .Arch import ARCH

class VGG(nn.Module):
    def __init__(self, name, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.name = name
        self.conv_layer_architecture = ARCH[name]
        self.conv_layers = self.create_conv_layers()
        self.conv_fcs = self.create_conv_fcs(num_classes)
        self._initialize_weights()

        self.epoch_loss_lis = None
        self.epoch_acc_lis = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_fcs(x)
        x = x.view(x.size(0), -1)  # Flatten for the final classification layer
        return x

    def create_conv_layers(self):
        layers = []
        in_channels = self.in_channels
        for x in self.conv_layer_architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)),
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif x == 'L':
                layers += [nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)]
        return nn.Sequential(*layers)

    def create_conv_fcs(self, num_classes):
        conv_fcs = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Ensure the feature map is 7x7
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=(1, 1)),

            nn.AdaptiveAvgPool2d((1, 1))  # output of size 1 x 1 x num_classes
        )
        return conv_fcs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)