"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        base_channels = self.hp.get("base_channels", 48)
        dropout = self.hp.get("dropout", 0.10)

        b = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            ConvLayer(3, b),
            ConvLayer(b, b),
        )

        self.enc2 = nn.Sequential(
            ConvLayer(b, 2 * b),
            ConvLayer(2 * b, 2 * b),
        )

        self.enc3 = nn.Sequential(
            ConvLayer(2 * b, 4 * b),
            ConvLayer(4 * b, 4 * b),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvLayer(4 * b, 8 * b),
            ConvLayer(8 * b, 8 * b),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

        # Decoder
        # After upsampling bottleneck: 8b
        # Skip from enc3: 4b
        # Concatenated: 12b
        self.dec3 = nn.Sequential(
            ConvLayer(12 * b, 4 * b),
            ConvLayer(4 * b, 4 * b),
        )

        # After upsampling dec3: 4b
        # Skip from enc2: 2b
        # Concatenated: 6b
        self.dec2 = nn.Sequential(
            ConvLayer(6 * b, 2 * b),
            ConvLayer(2 * b, 2 * b),
        )

        # After upsampling dec2: 2b
        # Skip from enc1: b
        # Concatenated: 3b
        self.dec1 = nn.Sequential(
            ConvLayer(3 * b, b),
            ConvLayer(b, b),
        )

        # Classifier output shape: (B, num_classes, H, W)
        self.classifier = nn.Conv2d(b, num_classes, kernel_size=1)

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

        # Encoder
        x1 = self.enc1(x)                 # (B, b, H, W)
        x2 = self.enc2(self.pool(x1))     # (B, 2b, H/2, W/2)
        x3 = self.enc3(self.pool(x2))     # (B, 4b, H/4, W/4)

        # Bottleneck
        x4 = self.bottleneck(self.pool(x3))   # (B, 8b, H/8, W/8)
        x4 = self.dropout(x4)

        # Decoder stage 3
        x = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = self.dropout(x)

        # Decoder stage 2
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        # Decoder stage 1
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        # Pixel-wise logits
        x = self.classifier(x)

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