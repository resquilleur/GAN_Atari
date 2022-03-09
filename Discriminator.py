import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape, discr_filters):
        super(Discriminator, self).__init__()
        self.DISCR_FILTERS = discr_filters
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=self.DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(in_channels=self.DISCR_FILTERS, out_channels=self.DISCR_FILTERS * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.DISCR_FILTERS * 2),
            nn.Mish(),
            nn.Conv2d(in_channels=self.DISCR_FILTERS * 2, out_channels=self.DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.DISCR_FILTERS * 4),
            nn.Mish(),
            nn.Conv2d(in_channels=self.DISCR_FILTERS * 4, out_channels=self.DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.DISCR_FILTERS * 8),
            nn.Mish(),
            nn.Conv2d(in_channels=self.DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)
