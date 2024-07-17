from src.models.components.backbone.resnet import ResNet101

def build_backbone(pretrained=True):
    return ResNet101(pretrained=pretrained)