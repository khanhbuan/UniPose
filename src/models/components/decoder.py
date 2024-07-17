import torch
import torch.nn as nn
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, num_classes=14):
        super(decoder, self).__init__()
        low_level_inplanes=256

        self.conv1 = nn.Conv2d(in_channels=low_level_inplanes, out_channels=48,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=304, out_channels=256, kernel_size=3, 
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, 
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.1),
                                       nn.Conv2d(in_channels=256, out_channels=num_classes+1, 
                                                 kernel_size=1, stride=1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)     # (48, 92, 92)
        low_level_feat = self.bn1(low_level_feat)       # (48, 92, 92)
        low_level_feat = self.relu(low_level_feat)      # (48, 92, 92)
        low_level_feat = self.maxpool(low_level_feat)   # (48, 46, 46)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)   # (256, 46, 46)

        x = torch.cat((x, low_level_feat), dim=1) # (304, 46, 46)
        x = self.last_conv(x)   # (15, 46, 46)

        return x
    
def build_decoder(num_classes=14):
    model = decoder(num_classes=num_classes)
    return model

if __name__ == "__main__":
    decoder = build_decoder()
    x, feat = torch.rand(1, 256, 23, 23), torch.rand(1, 256, 92, 92)
    y = decoder(x, feat)
    print(y.shape)