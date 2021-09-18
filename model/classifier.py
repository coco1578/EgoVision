import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, num_classes):

        super(Classifier, self).__init__()

        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=num_classes)

    def forward(self, x):

        x = self.model(x)
        return x
