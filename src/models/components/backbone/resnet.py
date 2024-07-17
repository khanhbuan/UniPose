import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, 
                 dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes*4,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        layers = [3, 4, 23]
        blocks = [1, 2, 4]
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
        planes = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes[0], layers[0], strides[0], dilations[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], strides[1], dilations[1])
        self.layer3 = self._make_layer(block, planes[2], layers[2], strides[2], dilations[2])
        self.layer4 = self._make_MG_unit(block, planes[3], blocks, strides[3], dilations[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, num_layers, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_layers):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 
                            dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        x = self.conv1(input)   # (64, 184, 184)
        x = self.bn1(x)         # (64, 184, 184)
        x = self.relu(x)        # (64, 184, 184)
        x = self.maxpool(x)     # (64, 92, 92)

        x = self.layer1(x)      # (256, 92, 92)
        low_level_feat = x      # (256, 92, 92)
        x = self.layer2(x)      # (512, 46, 46)
        x = self.layer3(x)      # (1024, 23, 23)
        x = self.layer4(x)      # (2048, 23, 23)
        return x, low_level_feat

def ResNet101(pretrained=True):
    model = ResNet(Bottleneck, pretrained=pretrained)
    return model