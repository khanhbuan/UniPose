import torch.nn as nn
from src.models.components.backbone import build_backbone
from src.models.components.wasp import build_wasp
from src.models.components.decoder import build_decoder

class unipose(nn.Module):
    def __init__(self, num_classes=14):
        super(unipose, self).__init__()
        self.backbone = build_backbone(pretrained=True)
        self.wasp = build_wasp()
        self.decoder = build_decoder(num_classes=num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        
        return x