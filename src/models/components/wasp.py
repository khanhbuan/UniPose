import torch
import torch.nn.functional as F
import torch.nn as nn

class _AtrousModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                     kernel_size=kernel_size, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        
        return self.relu(x)

class wasp(nn.Module):
    def __init__(self):
        super(wasp, self).__init__()
        inplanes = 2048
        dilations = [24, 18, 12, 6]

        self.aspp1 = _AtrousModule(inplanes=inplanes, planes=256, kernel_size=1, 
                                   padding=0, dilation=dilations[0])
        self.aspp2 = _AtrousModule(inplanes=256, planes=256, kernel_size=3, 
                                   padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _AtrousModule(inplanes=256, planes=256, kernel_size=3, 
                                   padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _AtrousModule(inplanes=256, planes=256, kernel_size=3, 
                                   padding=dilations[3], dilation=dilations[3])
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                             nn.Conv2d(in_channels=inplanes, out_channels=256, 
                                                       kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(num_features=256),
                                             nn.ReLU())
        
        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=256, 
                               kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)  # (256, 23, 23)
        x2 = self.aspp2(x1) # (256, 23, 23)
        x3 = self.aspp3(x2) # (256, 23, 23)
        x4 = self.aspp4(x3) # (256, 23, 23)

        x1 = self.conv2(x1) # (256, 23, 23)
        x2 = self.conv2(x2) # (256, 23, 23)
        x3 = self.conv2(x3) # (256, 23, 23)
        x4 = self.conv2(x4) # (256, 23, 23)

        x1 = self.conv2(x1) # (256, 23, 23)
        x2 = self.conv2(x2) # (256, 23, 23)
        x3 = self.conv2(x3) # (256, 23, 23)
        x4 = self.conv2(x4) # (256, 23, 23)

        x5 = self.global_avg_pool(x) # (256, 1, 1)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1) # (1280, 23, 23)

        x = self.conv1(x)   # (256, 23, 23)
        x = self.bn1(x)     # (256, 23, 23)
        x = self.relu(x)    # (256, 23, 23)

        return self.dropout(x)

def build_wasp():
    model = wasp()
    return model

if __name__ == "__main__":
    wasp = build_wasp().eval()
    input = torch.rand(1, 2048, 23, 23)
    output = wasp(input)
    print(output.shape)