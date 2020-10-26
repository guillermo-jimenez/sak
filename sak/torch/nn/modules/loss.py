from typing import Tuple, Iterable
import numpy as np
import torch 
import torch.nn
import sak
from torch.nn import Conv1d
from torch.nn import MSELoss
from torch.nn import BCELoss

from sak.__ops import required
from sak.__ops import check_required

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
            operation_class = sak.class_selector(operation['class'])
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
            self.reduction = sak.class_selector(reduction)

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
    def __init__(self, reduction: str = 'mean', eps: float = 1e-6, weight: torch.Tensor = None):
        # Epsilon (division by zero)
        self.eps = eps
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]

        # Reduction
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x

        super().__init__()
        
    def forward(self, input: torch.tensor, target: torch.tensor, sample_weight: torch.tensor = None) -> torch.tensor:
        # Preprocess inputs
        input = torch.flatten(input, start_dim=2)
        target = torch.flatten(target, start_dim=2)

        # Compute dice loss (per sample)
        intersection = (target*input).sum(-1)
        union = (target+input).sum(-1)
        
        # Apply class weights (https://arxiv.org/pdf/1707.03237.pdf)
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            self.weight = self.weight.to(target.device)
            intersection = (intersection*self.weight)
            union = (union*self.weight)
            
        # Average over channels
        intersection = intersection.sum(-1)
        union = union.sum(-1)
        
        # Compute loss
        loss = 1 - 2.*(intersection + self.eps)/(union + self.eps)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Return reduced (batch) loss
        return self.reduction(loss)


class BoundDiceLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean', eps: float = 1e-6, weight: Iterable = None, kernel_size: int = 25):
        super().__init__()
        # Save inputs
        self.reduction = reduction
        self.eps = eps
        self.weight = weight
        self.kernel_size = kernel_size
        
        # Define auxiliary loss
        self.loss = DiceLoss(reduction,eps,weight)
        
        # Define convolutional operation
        self.conv_op = Conv1d(3,3,kernel_size,padding=(kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.conv_op.parameters():
            param.requires_grad = False

        # Override weight
        self.conv_op.weight[:,:,:] = 0.
        self.conv_op.weight[0,0,0] = -1.
        self.conv_op.weight[1,1,0] = -1.
        self.conv_op.weight[2,2,0] = -1.
        self.conv_op.weight[0,0,-1] = 1.
        self.conv_op.weight[1,1,-1] = 1.
        self.conv_op.weight[2,2,-1] = 1.

    
    def forward(self, input: torch.tensor, target: torch.tensor, sample_weight: torch.tensor = None):
        # Move operation to device
        self.conv_op = self.conv_op.to(target.device)

        # Retrieve boundaries
        boundary_input = self.conv_op(input).abs()
        boundary_target = self.conv_op(target).abs()

        # Obtain dice loss between produced boundary masks
        return self.loss(boundary_input, boundary_target, sample_weight)


class InstanceLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean', weight: Iterable = None):
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            if not isinstance(weight, torch.Tensor):
                self.weight = torch.tensor(weight)
            else:
                self.weight = weight
            if self.weight.dim() == 1:
                self.weight = self.weight[None,]
        if reduction == 'mean':   self.reduction = torch.mean
        elif reduction == 'sum':  self.reduction = torch.sum
        elif reduction == 'none': self.reduction = lambda x: x
        
        # Define auxiliary loss
        self.loss = MSELoss(reduction='none')
        
        # Define convolutional operation
        self.conv_op = Conv1d(3,3,3,padding=1,bias=False)
        
        # Mark as non-trainable
        for param in self.conv_op.parameters():
            param.requires_grad = False

        # Override weight
        self.conv_op.weight[:,:,:] = 0.
        self.conv_op.weight[0,0,0] = -1.
        self.conv_op.weight[1,1,0] = -1.
        self.conv_op.weight[2,2,0] = -1.
        self.conv_op.weight[0,0,-1] = 1.
        self.conv_op.weight[1,1,-1] = 1.
        self.conv_op.weight[2,2,-1] = 1.

    
    def forward(self, input: torch.tensor, target: torch.tensor, sample_weight: torch.tensor = None):
        # Move operation to device
        self.conv_op = self.conv_op.to(target.device)

        # Retrieve boundaries
        boundary_input = self.conv_op(input).abs()
        boundary_target = self.conv_op(target).abs()

        # Sum of elements alongside the spatial dimensions
        boundary_input = torch.flatten(boundary_input, start_dim=2).sum(-1)/2
        boundary_target = torch.flatten(boundary_target, start_dim=2).sum(-1)/2

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            boundary_input = boundary_input*self.weight
            boundary_target = boundary_target*self.weight

        # Obtain per-sample loss
        loss = self.loss(boundary_input, boundary_target)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)


# class KLDivergence:
#     def __init__(self, reduction='mean', batch_size=required, input_shape=required, **kwargs):
#         # Mimic pytorch reductions (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py#L373)
#         self.batch_size = batch_size
#         self.input_shape = input_shape

#         reduction = reduction.lower()
#         if reduction in ['sum', 'mean']:
#             self.reduction = sak.class_selector('torch',reduction)
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
