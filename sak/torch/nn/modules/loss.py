from typing import Union, Tuple, Iterable, Callable, Dict, List
import numpy as np
import torch 
import torch.nn
import sak
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import Sigmoid
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import BCELoss

from sak.__ops import required
from sak.__ops import check_required

class none:
    def __call__(self, *args,**kwargs): # Stupid wrapper to homogeinize code with the imported classes
        return 0

class CompoundLoss(torch.nn.Module):
    def __init__(self, operations: Iterable[Union[Callable, Dict]], weight: Iterable[Union[int, float, Dict]] = None):
        super().__init__()
        self.operations = []
        for op in operations:
            if isinstance(op, Callable):
                self.operations.append(op)
            elif isinstance(op, dict):
                assert "class" in op,     "Missing 'class' field in dictionary"
                assert "arguments" in op, "Missing 'arguments' field in dictionary"
                # Retrieve operation class
                self.operations.append(sak.from_dict(op))
            else:
                raise ValueError("Invalid inputs provided of type {}".format(type(op)))
        
        # Store weights
        self.weight = []
        if weight is None:
            self.weight = [1]*len(self.operations)
        else:
            for w in weight:
                if isinstance(w, int) or isinstance(w, float):
                    self.weight.append(w)
                elif isinstance(w, dict):
                    self.weight.append(sak.from_dict(w))
                else:
                    raise ValueError("Invalid inputs provided of type {}".format(type(w)))

        # Assert number of weights is the same as number of operations
        assert len(self.operations) == len(self.weight), "The number of provided operations mismatches the size of the provided weights"
            
    def forward(self, state={}, **kwargs) -> torch.Tensor:
        loss = 0

        for i in range(len(self.operations)):
            # add loss to total loss
            w = self.weight[i]
            # If loss vary given the epoch
            if isinstance(self.weight[i], Iterable):
                if "epoch" in state:
                    w = w[min([state["epoch"],len(w)-1])]
                else:
                    w = w[-1]
            loss += w*self.operations[i](**kwargs)
            
        return loss


class PearsonCorrelationLoss: # Stupid wrapper to homogeinize code with the imported classes
    def __init__(self):
        pass

    def __call__(self, X_pred: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739"""
        vX_pred = X_pred - torch.mean(X_pred)
        vX = X - torch.mean(X)

        return torch.sum(vX_pred * vX) / (torch.sqrt(torch.sum(vX_pred ** 2)) * torch.sqrt(torch.sum(vX ** 2)))

class ConstantLoss:
    def __init__(self, value: Union[float, torch.Tensor] = 1.0):
        self.value = torch.tensor(value)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.value

class KLD_MSE:
    def __init__(self, reduction: str = 'mean', beta: float = 0.01, **kwargs):
        self.reduction = reduction
        self.beta = beta

        self.MSELoss = MSELoss(reduction=self.reduction)
        self.KLDivergence = KLDivergence(reduction=self.reduction, **kwargs)

    def __call__(self, X_pred: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mse = self.MSELoss(X_pred, X)
        kl = self.KLDivergence(mu, logvar)

        return mse + self.beta*kl

class KLD_BCE:
    def __init__(self, reduction: str = 'mean', beta: float = 0.01, **kwargs):
        self.reduction = reduction
        self.beta = beta

        self.BCELoss = BCELoss(reduction=self.reduction)
        self.KLDivergence = KLDivergence(reduction=self.reduction, **kwargs)

    def __call__(self, X_pred: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        bce = self.BCELoss(X_pred, X)
        kl  = self.KLDivergence(mu, logvar)

        return bce + self.beta*kl

class KLDivergence:
    def __init__(self, reduction: str = 'torch.mean', **kwargs):
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
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None) -> torch.tensor:
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



class BoundDiceLoss1d(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', eps: float = 1e-6, weight: Iterable = None, kernel_size: int = 1):
        super().__init__()
        self.register_buffer('weight', weight)

        # Save inputs
        self.channels = channels
        self.reduction = reduction
        self.eps = eps
        self.weight = weight
        self.kernel_size = kernel_size
        
        # Define auxiliary loss
        self.loss = DiceLoss(reduction,eps,weight)
        
        # Define convolutional operation
        self.conv_op = Conv1d(self.channels,self.channels,kernel_size,padding=(kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.conv_op.parameters():
            param.requires_grad = False

        # Override weight
        self.conv_op.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.conv_op.weight[c,c, 0] = -1.
            self.conv_op.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.conv_op = self.conv_op.to(target.device)

        # Retrieve boundaries
        boundary_input = self.conv_op(input).abs()
        boundary_target = self.conv_op(target).abs()

        # Obtain dice loss between produced boundary masks
        return self.loss(boundary_input, boundary_target, sample_weight)


class BoundDiceLoss2d(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', eps: float = 1e-6, weight: Iterable = None, kernel_size: int = 1):
        super().__init__()
        self.register_buffer('weight', weight)

        # Save inputs
        self.channels = channels
        self.reduction = reduction
        self.eps = eps
        self.weight = weight
        self.kernel_size = kernel_size
        
        # Define auxiliary loss
        self.loss = DiceLoss(reduction,eps,weight)
        
        # Define convolutional operation
        self.prewittx  = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        self.prewitty  = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        self.prewittxy = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        self.prewittyx = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)

        # Mark as non-trainable
        for param in self.prewittx.parameters(): param.requires_grad = False
        for param in self.prewitty.parameters(): param.requires_grad = False
        for param in self.prewittxy.parameters(): param.requires_grad = False
        for param in self.prewittyx.parameters(): param.requires_grad = False

        # Override weights to make Prewitt filters
        self.prewittx.weight[...]  = 0.
        self.prewitty.weight[...]  = 0.
        self.prewittxy.weight[...] = 0.
        self.prewittyx.weight[...] = 0.
        for c in range(self.channels):
            # x
            self.prewittx.weight[c,c, 0, 0]  = -1.
            self.prewittx.weight[c,c,-1, 0]  =  1.
            # y
            self.prewitty.weight[c,c, 0, 0]  = -1.
            self.prewitty.weight[c,c, 0,-1]  =  1.
            # xy
            self.prewittxy.weight[c,c, 0, 0] = -1.
            self.prewittxy.weight[c,c,-1,-1] =  1.
            # yx
            self.prewittyx.weight[c,c,-1, 0] =  1.
            self.prewittyx.weight[c,c, 0,-1] = -1.
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewittx   = self.prewittx.to(target.device)
        self.prewitty   = self.prewitty.to(target.device)
        self.prewittxy  = self.prewittxy.to(target.device)
        self.prewittyx  = self.prewittyx.to(target.device)

        # Obtain number of structures of the input
        input_bound_x   = self.prewittx(input).abs()
        input_bound_y   = self.prewitty(input).abs()
        input_bound_xy  = self.prewittxy(input).abs()
        input_bound_yx  = self.prewittyx(input).abs()

        # Obtain number of structures of the target
        target_bound_x  = self.prewittx(target).abs()
        target_bound_y  = self.prewitty(target).abs()
        target_bound_xy = self.prewittxy(target).abs()
        target_bound_yx = self.prewittyx(target).abs()
        
        # Boundary loss
        loss_x  = self.loss(input_bound_x,target_bound_x,sample_weight)
        loss_y  = self.loss(input_bound_y,target_bound_y,sample_weight)
        loss_xy = self.loss(input_bound_xy,target_bound_xy,sample_weight)
        loss_yx = self.loss(input_bound_yx,target_bound_yx,sample_weight)

        return (loss_x+loss_y+loss_xy+loss_yx)/4


class F1InstanceLoss1d(torch.nn.Module):
    def __init__(self, channels: int = 1, reduction: str = 'mean', weight: Iterable = None, kernel_size: int = 3):
        super().__init__()
        self.register_buffer('weight', weight)

        self.channels = channels
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
        self.kernel_size = kernel_size
        
        # Define convolutional operation
        self.prewitt = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        
        # Mark as non-trainable
        for param in self.prewitt.parameters():
            param.requires_grad = False

        # Override weights
        self.prewitt.weight[:,:,:] = 0.
        for c in range(self.channels):
            self.prewitt.weight[c,c, 0] = -1.
            self.prewitt.weight[c,c,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt = self.prewitt.to(target.device)

        # Retrieve boundaries
        input_boundary = self.prewitt(input).abs()
        target_boundary = self.prewitt(target).abs()

        # Sum of elements alongside the spatial dimensions
        input_elements = input_boundary.sum(-1)/4
        target_elements = target_boundary.sum(-1)/4

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            self.weight = self.weight.to(target.device)
            input_elements = input_elements*self.weight
            target_elements = target_elements*self.weight

        # F1 loss
        truepositive  = (target_elements-(target_elements-input_elements).clamp_min(0)).abs()
        falsepositive = (input_elements-target_elements).clamp_min(0)
        falsenegative = (target_elements-input_elements).clamp_min(0)

        # F1 loss
        loss = 1-(2*truepositive + 1)/(2*truepositive + falsepositive + falsenegative + 1)

        # Sum over channels
        loss = loss.sum(-1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain loss
        return self.reduction(loss)


class F1InstanceLoss2d(torch.nn.Module):
    def __init__(self, channels: int = 1, weight: Iterable = None, reduction: str = 'mean', kernel_size: int = 3):
        super().__init__()
        self.register_buffer('weight', weight)

        # Save inputs
        self.channels = channels
        if weight is None:
            self.weight = torch.ones((1,self.channels))
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

        # Check weights size
        assert self.weight.size(-1) == channels, "The number of provided channels and the associated weights do not match"
        
        # Define auxiliary loss
        self.kernel_size = kernel_size
        
        # Define convolutional operation
        self.prewitt  = Conv1d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        self.prewittx = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)
        self.prewitty = Conv2d(self.channels,self.channels,self.kernel_size,padding=(self.kernel_size-1)//2,bias=False)

        # Mark as non-trainable
        for param in self.prewitt.parameters():  param.requires_grad = False
        for param in self.prewittx.parameters(): param.requires_grad = False
        for param in self.prewitty.parameters(): param.requires_grad = False

        # Override weights to make Prewitt filters
        self.prewitt.weight[...]  = 0.
        self.prewittx.weight[...] = 0.
        self.prewitty.weight[...] = 0.
        for c in range(self.channels):
            # border
            self.prewitt.weight[c,c, 0]    = -1.
            self.prewitt.weight[c,c,-1]    =  1.
            # x
            self.prewittx.weight[c,c, 0,0] = -1.
            self.prewittx.weight[c,c,-1,0] =  1.
            # y
            self.prewitty.weight[c,c,0, 0] = -1.
            self.prewitty.weight[c,c,0,-1] =  1.

    
    def forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor = None):
        # Move operation to device
        self.prewitt  =  self.prewitt.to(target.device)
        self.prewittx = self.prewittx.to(target.device)
        self.prewitty = self.prewitty.to(target.device)
        
        # Obtain number of structures of the target
        target_bound_x    = self.prewittx(target).abs()
        target_bound_x_1d = self.prewitt(target_bound_x.sum(-2)).abs()
        target_elements_x = torch.flatten(target_bound_x_1d, start_dim=2).sum(-1)/(4**2)
        
        target_bound_y    = self.prewitty(target).abs()
        target_bound_y_1d = self.prewitt(target_bound_y.sum(-1)).abs()
        target_elements_y = torch.flatten(target_bound_y_1d, start_dim=2).sum(-1)/(4**2)
        
        # Obtain number of structures of the input
        input_bound_x     = self.prewittx(input).abs()
        input_bound_x_1d  = self.prewitt(input_bound_x.sum(-2)).abs()
        input_elements_x  = torch.flatten(input_bound_x_1d, start_dim=2).sum(-1)/(4**2)
        
        input_bound_y     = self.prewitty(input).abs()
        input_bound_y_1d  = self.prewitt(input_bound_y.sum(-1)).abs()
        input_elements_y  = torch.flatten(input_bound_y_1d, start_dim=2).sum(-1)/(4**2)

        # Apply class weights
        if self.weight is not None:
            # Assert compatible shapes
            assert self.weight.shape[-1] == input.shape[1], "The number of channels and provided class weights does not coincide"
            self.weight = self.weight.to(target.device)
            input_elements_x  =  input_elements_x*self.weight
            input_elements_y  =  input_elements_y*self.weight
            target_elements_x = target_elements_x*self.weight
            target_elements_y = target_elements_y*self.weight
        
        # F1 loss
        truepositive_x  = (target_elements_x-(target_elements_x-input_elements_x).clamp_min(0)).abs()
        falsepositive_x = (input_elements_x-target_elements_x).clamp_min(0)
        falsenegative_x = (target_elements_x-input_elements_x).clamp_min(0)

        truepositive_y  = (target_elements_y-(target_elements_y-input_elements_y).clamp_min(0)).abs()
        falsepositive_y = (input_elements_y-target_elements_y).clamp_min(0)
        falsenegative_y = (target_elements_y-input_elements_y).clamp_min(0)

        # F1 loss
        loss_x = 1-(2*truepositive_x + 1)/(2*truepositive_x + falsepositive_x + falsenegative_x + 1)
        loss_y = 1-(2*truepositive_y + 1)/(2*truepositive_y + falsepositive_y + falsenegative_y + 1)
        loss   = (loss_x+loss_y)/2

        # Sum over channels
        loss = loss.sum(-1)

        # Apply sample weight to samples
        if sample_weight is not None:
            loss *= sample_weight

        # Obtain per-sample loss
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

