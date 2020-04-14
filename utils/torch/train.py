from typing import Any
from typing import Tuple
from typing import List
from typing import Callable
import os
import os.path
import shutil
import tqdm
import torch
import torch.nn
import numpy as np
import utils.torch.data

def do_epoch(model: torch.nn.Module, state: dict, execution: dict, dataloader: torch.utils.data.DataLoader, criterion: Callable, metric: Callable = None) -> Tuple[list, float]:
    """
    Minimum do_epoch example
    1. Select device to send tensors
    2. Initialize loss function
    3. Predict + optimize batch
    4. Save loss per batch (useful given size of dataset)
    """
    
    # Record progress
    batch_loss = np.zeros((len(dataloader),),dtype='float16')

    # Apply data augmentation
    if 'augmentation' in execution:
        transforms = []
        for k in execution['augmentation']['types']:
            transforms.append(utils.class_selector('utils.torch.data.augmentation',k)(*execution['augmentation']['types'][k]))
            
        augmentation = utils.class_selector('torchvision.transforms',execution['augmentation']['name'])(transforms, *execution['augmentation']['arguments'])

    # Select iterator decorator
    train_type = 'Train' if model.training else 'Valid'
    iterator = utils.get_tqdm(dataloader, execution['iterator'], desc="({}) Epoch {:>3d}/{:>3d}, Loss {:0.3f}".format(train_type, state['epoch']+1, execution['epochs'], np.inf))

    # Iterate over all data in train/validation/test dataloader:
    print_loss = np.inf
    loss = 0
    for i, (X, y) in enumerate(iterator):
        # # Apply data augmentation
        if model.training and ('augmentation' in execution):
            X = augmentation(X)

        # Send elements to device
        X = X.float().to(state['device'], non_blocking=True)
        y = y.float().to(state['device'], non_blocking=True)

        # Set gradient to zero
        if model.training: 
            state['optimizer'].zero_grad()

        # Predict input data
        out = model(X)
        if not isinstance(out, tuple):
            out = (out,)

        # Calculate loss
        loss = criterion(X,y,*out)

        # Break early
        if torch.isnan(loss):
            raise ValueError("Nan loss value encountered. Stopping...")

        # Retrieve for printing purposes
        print_loss = metric(X,y,*out) if metric is not None else loss.item()
        
        # Optimize network's weights
        if model.training:
            loss.backward()
            state['optimizer'].step()

        # Accumulate losses
        batch_loss[i] += print_loss

        # Change iterator description
        if isinstance(iterator,tqdm.tqdm):
            iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}".format(train_type, state['epoch']+1, execution['epochs'], print_loss))
        
    if isinstance(iterator, tqdm.tqdm):
        iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}".format(train_type, state['epoch']+1, execution['epochs'], np.mean(batch_loss)))

    return batch_loss, np.sum(batch_loss)/dataloader.batch_size


def train_model(model, state: dict, execution: dict, loader_train: torch.utils.data.DataLoader, loader_valid: torch.utils.data.DataLoader, criterion: Callable, metric: Callable = None):
    model = model.to(state['device'])

    epoch_train = []
    epoch_valid = []

    for epoch in range(state['epoch'], execution['epochs']):
        try:
            # Store current epoch
            state['epoch'] = epoch
            
            # Training model
            loss_train, state['train_loss'] = do_epoch(model.train(), state, execution, loader_train, criterion, metric)
            epoch_train.append(loss_train)

            # Validate results
            loss_valid, state['validation_loss'] = do_epoch(model.eval(), state, execution, loader_valid, criterion, metric)
            epoch_valid.append(loss_valid)

            # Update learning rate scheduler
            if 'scheduler' in state:
                state['scheduler'].step(np.mean(loss_valid))
                # state['scheduler'].step(state['validation_loss'])

            # Checkpoint model when best performance
            state['model_state_dict'] = model.state_dict()
            
            # Save model info
            utils.pickledump(state, os.path.join(execution['save_directory'],'checkpoint.state'), mode='wb')
            
            # Check if loss is best loss
            if state['validation_loss'] < state['best_loss']:
                state['best_loss'] = state['validation_loss']
                state['best_epoch'] = epoch
                
                # Copy checkpoint and mark as best
                shutil.copyfile(os.path.join(execution['save_directory'],'checkpoint.state'), os.path.join(execution['save_directory'],'model_best.state'))
            
            # # Apply early stopping if scheduler not being used
            # if ('scheduler' not in state) and ('patience' in execution):
            #     if epoch-state['best_epoch'] >= execution['patience']:
            #         print(" * No improvement in {} epochs, stopping".format(patience))
            #         break
        except KeyboardInterrupt:
            utils.pickledump(state, os.path.join(execution['save_directory'],'keyboard_interrupt.state'), mode='wb')
            raise
        except:
            utils.pickledump(state, os.path.join(execution['save_directory'],'error.state'), mode='wb')
            raise
            
