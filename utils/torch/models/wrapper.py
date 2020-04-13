import torch
import torch.nn
import utils.torch.nn
from typing import Tuple

# Check required arguments as keywords
from utils.__ops import required
from utils.__ops import check_required

class VariationalMLPClassification(torch.nn.Module):
    def __init__(self, VAE: utils.torch.models.Variational, classes: int = required, 
                       linear_neurons: list = required):
        super(VariationalMLPClassification, self).__init__()

        # Check inputs
        in_dict = {
            'VAE'            : VAE,
            'classes'        : classes,
            'linear_neurons' : linear_neurons,
        }
        check_required(self, in_dict)

        # Store inputs
        self.VAE = VAE
        self.latent_dimension = self.VAE.latent_dimension
        self.linear_neurons = linear_neurons

        # Set operations
        initial_operation = [
            utils.torch.nn.Linear(self.latent_dimension, self.linear_neurons[0]),
        ]
        middle_operations = []
        for i in range(len(self.linear_neurons)-1):
            # # Activation
            middle_operations.append(utils.torch.nn.ReLU())

            # Dropout
            middle_operations.append(utils.torch.nn.Dropout(0.25))

            # Operation
            middle_operations.append(utils.torch.nn.Linear(self.linear_neurons[i], self.linear_neurons[i+1]))

        final_operation   = [
            utils.torch.nn.ReLU(),
            # utils.torch.nn.Dropout(0.25),
            utils.torch.nn.Linear(self.linear_neurons[-1], classes),
            # utils.torch.nn.Softmax(dim=-1)
        ]

        # Define classification operations
        self.linear_operations = utils.torch.nn.Sequential(*initial_operation, *middle_operations, *final_operation)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z, mu, logvar = self.VAE.encode(x)
        y = self.linear_operations(z)
        xhat = self.VAE.decode(z)
        return xhat, y, mu, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.VAE.encode(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.VAE.decode(z)

    
