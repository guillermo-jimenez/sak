import torch
import torch.nn
import utils.modules
import utils.modules.blocks
from typing import *

class ConvolutionalLatentMapper(torch.nn.Module):
    def __init__(self, in_shape: torch.Size, latent_neurons: int, preactivation: bool = True, dropout_rate: float = 0.25, **kwargs: dict):
        super(ConvolutionalLatentMapper, self).__init__()
        
        # Pre-activation
        self.in_shape = in_shape
        self.preactivation = preactivation
        self.latent_neurons = latent_neurons
        self.dropout_rate = dropout_rate

        if self.preactivation:
            self.activation = torch.nn.PReLU() # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
            self.normalization = torch.nn.BatchNorm1d(in_filt)
            self.dropout = utils.modules.SpatialDropout1d(self.dropout_rate)
        
        self.mu_mapper = utils.modules.blocks.Linear()


    def forward(self, x: torch.Tensor):
        if self.preactivation:
            h = self.activation(x)
            h = self.normalization(h)
            h = self.dropout(h)

        h = self.separable_conv1(x)
        h = self.separable_conv2(h)
        
        # If the number of channels of x and h does not coincide,
        # apply same transformation to x
        if x.shape[1] != h.shape[1]:
            x = self.pointwise_conv(x)

        return x + h # Residual connection


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 img_size: int):
        super(EncoderBlock, self).__init__()

        # Build Encoder
        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU())

        out_size = conv_out_shape(img_size)
        self.encoder_mu = nn.Linear(out_channels * out_size ** 2 , latent_dim)
        self.encoder_var = nn.Linear(out_channels * out_size ** 2, latent_dim)

    def forward(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        h = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encoder_mu(h)
        log_var = self.encoder_var(h)

        return [result, mu, log_var]