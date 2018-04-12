import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(kernel_size, stride, in_size, out_size ):
    layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2)),
            #nn.BatchNorm2d(out_size, momentum=0.01, eps=0.001),
            nn.BatchNorm2d(out_size),
            nn.MaxPool2d(kernel_size=2, padding=1),
            #nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )
    return layer

class Model(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes):
        super(Model, self).__init__()
        num_filters = 32
        kernel_size = 3
        dims = 84 # Mini 

        self.layer1 = conv(kernel_size, 1, 3, num_filters)
        self.layer2 = conv(kernel_size, 1, num_filters, num_filters) 
        self.layer3 = conv(kernel_size, 1, num_filters, num_filters)
        self.layer4 = conv(kernel_size, 1, num_filters, num_filters)

        self.final = nn.Linear(num_filters * 7 * 7, num_classes)
        #self.final = nn.Linear(num_filters * 5 * 5, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        return self.final(x)

    def embedding(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x.view(x.size(0), -1)

