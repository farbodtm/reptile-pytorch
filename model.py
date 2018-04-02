import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(kernel_size, stride, in_size, out_size ):
    layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_size),
            nn.MaxPool2d(kernel_size=2, padding=(out_size/4)),
            nn.ReLU()
            )
    return layer

class Model(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes, learning_rate):
        super(Model, self).__init__()
        num_filters = 32
        kernel_size = 3
        dims = 84 # Mini 

        self.layers = [conv(kernel_size, 1, 3, num_filters)]
        for _ in range(3):
            self.layers.append(conv(3, 1, num_filters, num_filters))

        self.final = nn.Linear(num_filters * dims * dims, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        return self.final(x)
