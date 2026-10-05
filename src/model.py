import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


NUM_CLASSES = 6


def create_model(pretrained=True):
    if pretrained:
        weights = ResNet34_Weights.DEFAULT
    else:
        weights = None

    model = resnet34(weights=weights)

    # Replace the original ImageNet classification layer
    # with our 6 inventory-count classes.
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


def predict(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return prediction, confidence, probabilities