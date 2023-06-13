import torch
import torchvision
import torch.nn as nn


class MobileNetAntispoof(torch.nn.Module):
    def __init__(self):
        super(MobileNetAntispoof, self).__init__()

        self.model = torchvision.models.mobilenet_v3_large()
        self.model.classifier[3] = nn.Linear(
            in_features=1280, out_features=2, bias=True
        )
        self.model.classifier.append(module=nn.Sigmoid())
        self.model.load_state_dict(
            torch.load("antispoof\models\mobilenet", map_location=torch.device("cpu"))
        )
        self.model.eval()

    def forward(self, x):
        return self.model(x)
