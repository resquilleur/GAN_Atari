import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, output_shape, latent_vector, gener_filters):
        super(Generator, self).__init__()
        self.LATENT_VECTOR = latent_vector
        self.GENER_FILTERS = gener_filters
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.LATENT_VECTOR, out_channels=self.GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(self.GENER_FILTERS * 8),
            nn.Mish(),
            nn.ConvTranspose2d(in_channels=self.GENER_FILTERS*8, out_channels=self.GENER_FILTERS*4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.GENER_FILTERS*4),
            nn.Mish(),
            nn.ConvTranspose2d(in_channels=self.GENER_FILTERS*4, out_channels=self.GENER_FILTERS*2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.GENER_FILTERS*2),
            nn.Mish(),
            nn.ConvTranspose2d(in_channels=self.GENER_FILTERS*2, out_channels=self.GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.GENER_FILTERS),
            nn.Mish(),
            nn.ConvTranspose2d(in_channels=self.GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)
