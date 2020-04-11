from typing import *
import torch
import torch.nn
from abc import abstractmethod

class Variational(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Variational, self).__init__()

        # Enforce existence of latent dimension variable
        self.latent_dimension = latent_dimension


    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) ->  torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def summary(model: torch.nn.Module, file):
    with open(file, 'w') as f:
        f.write(model.__repr__())


