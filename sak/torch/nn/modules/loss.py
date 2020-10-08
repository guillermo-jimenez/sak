from typing import Tuple
import numpy as np
import torch 
import torch.nn
import utils
from torch.nn import MSELoss
from torch.nn import BCELoss

from utils.__ops import required
from utils.__ops import check_required

class none:
    def __call__(self, *args,**kwargs): # Stupid wrapper to homogeinize code with the imported classes
        return 0

class CompoundLoss:
    def __init__(self, json: dict):
        self.operations = []
        self.weights = []
        self.mappings = []
        
        for operation in json:
            # Retrieve operation class
            operation_class = utils.class_selector(operation['class'])
            # Append the instantiated operation alongside weights and mapping
            self.operations.append(operation_class(**operation.get('arguments',{})))
            self.weights.append(operation['weight'])
            self.mappings.append(operation['mapping'])
            
    def __call__(self, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for i in range(len(self.operations)):
            selected = []
            for m in self.mappings[i]:
                (type, index) = m.split('_')
                if type.lower() == 'input':
                    selected.append(inputs[int(index)])
                elif type.lower() == 'output':
                    selected.append(outputs[int(index)])
                else:
                    raise ValueError("Wrong JSON. Make this error msg better at some point")
            
            # add loss to total loss
            loss += self.weights[i]*self.operations[i](*selected)
            
        return loss

class PearsonCorrelationLoss: # Stupid wrapper to homogeinize code with the imported classes
    def __init__(self):
        pass

    def __call__(self, X_pred: torch.Tensor, X: torch.tensor) -> torch.Tensor:
        """https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739"""
        vX_pred = X_pred - torch.mean(X_pred)
        vX = X - torch.mean(X)

        return torch.sum(vX_pred * vX) / (torch.sqrt(torch.sum(vX_pred ** 2)) * torch.sqrt(torch.sum(vX ** 2)))

# def KLD_MSE(reduction='mean'):
#     def loss(X_pred: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         # BCE = torch.nn.functional.binary_cross_entropy(X_pred, X.view(-1, 1024), reduction='mean')
#         # BCE = torch.nn.functional.binary_cross_entropy(X_pred, X.view(-1, 1024), reduction='mean')
#         # BCE = torch.nn.functional.mse_loss(X_pred, X.view(-1, 1024), reduction='mean')
#         # BCE = torch.nn.functional.mse_loss(X_pred, X.view(-1, 1024), reduction='mean')

#         # see Appendix B from VAE paper:
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         return BCE + KLD
#     return loss

class KLD_MSE:
    def __init__(self, reduction='mean', beta=0.01, **kwargs):
        self.reduction = reduction
        self.beta = beta

        self.MSELoss = MSELoss(reduction=self.reduction)
        self.KLDivergence = KLDivergence(reduction=self.reduction, **kwargs)

    def __call__(self, X_pred: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mse = self.MSELoss(X_pred, X)
        kl = self.KLDivergence(mu, logvar)

        return mse + self.beta*kl

class KLD_BCE:
    def __init__(self, reduction='mean', beta=0.01, **kwargs):
        self.reduction = reduction
        self.beta = beta

        self.BCELoss = BCELoss(reduction=self.reduction)
        self.KLDivergence = KLDivergence(reduction=self.reduction, **kwargs)

    def __call__(self, X_pred: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        bce = self.BCELoss(X_pred, X)
        kl  = self.KLDivergence(mu, logvar)

        return bce + self.beta*kl

class KLDivergence:
    def __init__(self, reduction='torch.mean', **kwargs):
        # Mimic pytorch reductions (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py#L373)
        # Preprocess input
        reduction = reduction.lower()
        if len(reduction.split('.')) == 1: 
            reduction = 'torch.{}'.format(reduction)

        # Retrieve function
        if reduction == 'torch.none': 
            self.reduction = lambda x: x
        else:                         
            self.reduction = utils.class_selector(reduction)

        # check input
        check_required(self, self.__dict__)

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ Solution of KL(qφ(z)||pθ(z)), Gaussian case:

        Sum over the dimensionality of z
        KL = - 0.5 * sum(1 + log(σ^2) - µ^2 - σ^2)

        When using a recognition model qφ(z|x) then µ and σ are simply functions of x and the variational parameters φ

        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        """

        # Implemented mean reduced KL divergence for consistency.
        return self.reduction(torch.exp(logvar) + mu**2 - 1. - logvar)

class DiceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6, smooth: int = 1.):
        self.eps = eps
        self.smooth = smooth
        super().__init__()
        
    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        intersection = (target*input).sum()
        union = (target+input).sum()
        loss = (1 - ((2. * intersection + self.smooth + self.eps) / 
                        (union + self.smooth + self.eps)))
        return torch.mean(loss)

# class KLDivergence:
#     def __init__(self, reduction='mean', batch_size=required, input_shape=required, **kwargs):
#         # Mimic pytorch reductions (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py#L373)
#         self.batch_size = batch_size
#         self.input_shape = input_shape

#         reduction = reduction.lower()
#         if reduction in ['sum', 'mean']:
#             self.reduction = utils.class_selector('torch',reduction)
#         elif reduction == 'none':
#             self.reduction = lambda x: x
#         else:
#             raise ValueError("Invalid reduction method '{}'".format(reduction))

#         # check input
#         check_required(self, self.__dict__)

#     def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """ Solution of KL(qφ(z)||pθ(z)), Gaussian case:

#         Sum over the dimensionality of z
#         KL = - 0.5 * sum(1 + log(σ^2) - µ^2 - σ^2)

#         When using a recognition model qφ(z|x) then µ and σ are simply functions of x and the variational parameters φ

#         See Appendix B from VAE paper:
#         Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         https://arxiv.org/abs/1312.6114
#         """

#         # Implemented mean reduced KL divergence for consistency.
#         return self.reduction(torch.exp(logvar) + mu**2 - 1. - logvar)
#         # return reduction(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
#         # return torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)/(self.batch_size*self.input_shape)
