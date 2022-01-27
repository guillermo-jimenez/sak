import math
import torch
import torch.optim
import torch.optim.lr_scheduler

class CosineWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, 
                       num_wait_steps: int = 0, num_cycles: float = 0.5, last_epoch: int = -1):
        self.optimizer          = optimizer
        self.num_warmup_steps   = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_wait_steps     = num_wait_steps
        self.num_cycles         = num_cycles
        self.last_epoch         = last_epoch
        
        # Get scheduler and wrap over scheduler
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,self._lr_lambda, self.last_epoch)
        self.state_dict = self._scheduler.state_dict
        self.load_state_dict = self._scheduler.load_state_dict
        self.get_lr = self._scheduler.get_lr
        self.lr_lambdas = self._scheduler.lr_lambdas
        self.base_lrs = self._scheduler.base_lrs
        self.last_epoch = self._scheduler.last_epoch
        self._step_count = self._scheduler._step_count
        self.verbose = self._scheduler.verbose
        self._get_lr_called_within_step = self._scheduler._get_lr_called_within_step
        self._last_lr = self._scheduler._last_lr        
        
    def _lr_lambda(self, current_step):
        if current_step < self.num_wait_steps:
            return 0.0

        if current_step < self.num_warmup_steps + self.num_wait_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps + self.num_wait_steps))

        progress = float(current_step - self.num_warmup_steps - self.num_wait_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps - self.num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))
        
