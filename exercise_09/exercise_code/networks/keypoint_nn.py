"""Models for facial keypoint detection"""

import torch
import torch.nn as nn



class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architectures, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        channels = hparams.get("channels", [32, 64, 128, 256])
        conv_dropout = hparams.get("conv_dropout", 0.05)
        fc_dropout = hparams.get("fc_dropout", 0.10)
        hidden_dim = hparams.get("hidden_dim", 256)

        cnn_layers = []
        in_channels = 1

        for out_channels in channels:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2)
            ])

            if conv_dropout > 0:
                cnn_layers.append(nn.Dropout2d(conv_dropout))

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Automatically calculate the flattened feature dimension.
        # This makes the model flexible if you change channels or pooling depth.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 96, 96)
            dummy_output = self.cnn(dummy_input)
            feature_dim = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),

            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),

            nn.Linear(128, 30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.cnn(x)
        x = self.fc(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
