"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
        lraspp_mobilenet_v3_large,
        LRASPP_MobileNet_V3_Large_Weights,
    )

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        pretrained = self.hp.get("pretrained", True)

        # 1. Load pretrained LRASPP-MobileNetV3-Large
        if pretrained:
            weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            self.model = lraspp_mobilenet_v3_large(weights=weights)
        else:
            self.model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)

        # 2. Replace the original classifier with a new 23-class classifier.
        low_in_channels = self.model.classifier.low_classifier.in_channels
        high_in_channels = self.model.classifier.high_classifier.in_channels

        self.model.classifier.low_classifier = nn.Conv2d(
            low_in_channels,
            num_classes,
            kernel_size=1
        )

        self.model.classifier.high_classifier = nn.Conv2d(
            high_in_channels,
            num_classes,
            kernel_size=1
        )

        # 3. ImageNet normalization.
        self.use_imagenet_norm = self.hp.get("imagenet_norm", True)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #  
        ########################################################################

        if self.use_imagenet_norm:
            x = (x - self.mean) / self.std

        # torchvision segmentation models return a dict:
        # {"out": tensor}
        x = self.model(x)["out"]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")