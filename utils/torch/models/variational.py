from typing import Any, List, Tuple
import inspect
import torch
import torch.nn
import utils
import utils.torch.blocks
# import utils.modules

# Check required arguments as keywords
from utils.__ops import required
from utils.__ops import check_required

class CVAE1d(utils.torch.models.Variational):
    def __init__(self, input_channels: int = required,
                       input_shape: int = required,
                       encoder_channels: list = required,
                       mapper_mlp: list = required,
                       latent_dimension: int = required,
                       decoder_channels: list = [],
                       regularization: dict = {},
                       operation_encoder: str = 'SeparableConv1d',
                       operation_decoder: str = 'SeparableConv1d',
                       **kwargs: dict):
        # Pass arguments to parent
        super(CVAE1d, self).__init__(latent_dimension)

        # 0) Check inputs
        in_dict = {
            'input_channels'    : input_channels,
            'input_shape'       : input_shape,
            'encoder_channels'  : encoder_channels,
            'mapper_mlp'        : mapper_mlp,
            'latent_dimension'  : latent_dimension,
        }
        check_required(self, in_dict)

        # Check inputs
        if not decoder_channels:
            decoder_channels = encoder_channels[::-1]
        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = 3

        # 1) Store inputs
        # encoder channels for convolutional operations
        self.input_channels         = input_channels
        self.input_shape            = input_shape
        self.mapper_mlp             = mapper_mlp
        self.regularization         = regularization
        self.encoder_channels       = [self.input_channels] + encoder_channels
        self.decoder_channels       = decoder_channels + [self.input_channels]
        self.encoder_linear_neurons = [self.encoder_channels[-1]*self.input_shape] + self.mapper_mlp
        self.decoder_linear_neurons = [self.latent_dimension] + self.mapper_mlp[::-1] + [self.encoder_channels[-1]*self.input_shape]

        ########### Encoder ###########
        # Encoder convolutional operations
        encoder_operations = []
        for i in range(len(self.encoder_channels)-1):
            if (i != 0):
                encoder_operations.append(utils.class_selector('utils.torch.activation', 'ReLU')())
                # encoder_operations.append(utils.class_selector('utils.torch.normalization', 'BatchNorm1d')(self.encoder_channels[i]))
                encoder_operations.append(utils.class_selector('utils.torch.dropout', 'Dropout1d')(0.25))

            encoder_operations.append(
                utils.class_selector('utils.torch.blocks', operation_encoder)(
                    self.encoder_channels[i],
                    self.encoder_channels[i+1],
                    **kwargs
                )
            )
        
        encoder_operations.append(utils.torch.blocks.Flatten())

        # Encoder linear mapping operations
        for i in range(len(self.encoder_linear_neurons)-1):
            encoder_operations.append(utils.class_selector('utils.torch.activation', 'ReLU')())
            encoder_operations.append(utils.class_selector('utils.torch.dropout', 'Dropout')(0.25))

            encoder_operations.append(
                utils.torch.blocks.Linear(
                    self.encoder_linear_neurons[i], 
                    self.encoder_linear_neurons[i+1], 
                )
            )

        encoder_operations.append(utils.class_selector('utils.torch.activation', 'ReLU')())
        encoder_operations.append(utils.class_selector('utils.torch.dropout', 'Dropout')(0.25))

        # To sequential
        self.encoder_operations = torch.nn.Sequential(*encoder_operations)

        ########### Bottleneck ###########
        self.bottleneck_mu = utils.torch.blocks.Linear(
            self.encoder_linear_neurons[-1], 
            self.latent_dimension, 
        )
        self.bottleneck_logvar = utils.torch.blocks.Linear(
            self.encoder_linear_neurons[-1], 
            self.latent_dimension, 
        )
        
        ########### Decoder ###########
        # Decoder operations
        decoder_operations = []
        for i in range(len(self.decoder_linear_neurons)-1):
            if (i != 0):
                decoder_operations.append(utils.class_selector('utils.torch.activation', 'ReLU')())
                decoder_operations.append(utils.class_selector('utils.torch.dropout', 'Dropout')(0.25))
            
            decoder_operations.append(
                utils.torch.blocks.Linear(
                    self.decoder_linear_neurons[i], 
                    self.decoder_linear_neurons[i+1], 
                )
            )

        decoder_operations.append(utils.torch.blocks.UnFlatten([self.decoder_channels[0],input_shape]))

        for i in range(len(self.decoder_channels)-1):
            decoder_operations.append(utils.class_selector('utils.torch.activation', 'ReLU')())
            # decoder_operations.append(utils.class_selector('utils.torch.normalization', 'BatchNorm1d')(self.decoder_channels[i]))
            decoder_operations.append(utils.class_selector('utils.torch.dropout', 'Dropout1d')(0.25))

            decoder_operations.append(
                utils.class_selector('utils.torch.blocks', operation_decoder)(
                    self.decoder_channels[i], 
                    self.decoder_channels[i+1], 
                    **kwargs
                )
            )
        
        # To sequential
        self.decoder_operations = torch.nn.Sequential(*decoder_operations)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        z, mu, logvar = self.encode(x)
        xhat = self.decode(z)
        return xhat, mu, logvar
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        h = self.encoder_operations(x)
        mu = self.bottleneck_mu(h)
        logvar = self.bottleneck_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_operations(z)

    def __str__(self):
        """To do"""
        s = ''
        return s

    def __repl__(self):
        return self.__str__()


class VAE1d(utils.torch.models.Variational):
    """VAE model for an arbitrary number of levels"""

    def __init__(self, input_shape: int = required, encoder_neurons: list = required, 
                       latent_dimension: int = required, decoder_neurons: list = required):
        super(VAE1d, self).__init__(latent_dimension)
        
        # Check inputs
        in_dict = {
            'input_shape'      : input_shape,
            'encoder_neurons'  : encoder_neurons,
            'latent_dimension' : latent_dimension,
            'decoder_neurons'  : decoder_neurons,
        }
        check_required(self, in_dict)

        # Store input
        self.encoder_neurons    = [input_shape] + encoder_neurons
        self.decoder_neurons    = decoder_neurons + [input_shape]
        
        # Create as many linear layers as chosen
        self.encoder_levels     = [utils.torch.blocks.Linear(self.encoder_neurons[i], self.encoder_neurons[i+1]) for i in range(len(self.encoder_neurons-1))]
        self.bottleneck_mu      = torch.nn.Linear(self.encoder_neurons[-1], self.latent_dimension)
        self.bottleneck_logvar  = torch.nn.Linear(self.encoder_neurons[-1], self.latent_dimension)
        self.decoder_levels     = [utils.torch.blocks.Linear(self.decoder_neurons[i], self.decoder_neurons[i+1]) for i in range(len(self.decoder_neurons-1))]
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        
    def encode(self, x: torch.Tensor):
        h = x
        
        for i in range(len(self.encoder_levels)):
            h = self.encoder_levels[i](h)
        
        mu = self.bottleneck_mu(h)
        logvar = self.bottleneck_logvar(h)

        return mu, logvar
        
    def decode(self, x: torch.Tensor):
        h = x
        
        for i in range(len(self.decoder_levels)):
            h = self.decoder_levels[i](h)
        
        return h

    def __str__(self):
        """To do"""
        s = ''
        return s

    def __repl__(self):
        return self.__str__()



