from typing import Any
from typing import Tuple
from typing import List
from typing import Callable
import os
import dill
import os.path
import shutil
import tqdm
import torch
import torch.nn
import numpy as np
import sak.torch.data

def do_epoch(model: torch.nn.Module, state: dict, execution: dict, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: Callable) -> list:
    """
    Minimum do_epoch example
    1. Select device to send tensors
    2. Initialize loss function
    3. Predict + optimize batch
    4. Save loss per batch (useful given size of dataset)
    """
    
    # Record progress
    batch_loss = np.zeros((len(dataloader),),dtype='float16')

    # Create transforms
    if ('data_pre' in execution):
        data_pre = sak.class_selector(execution["data_pre"]["class"])(**execution["data_pre"]["arguments"])
    if ('augmentation' in execution) and model.training:
        augmentation = sak.class_selector(execution["augmentation"]["class"])(**execution["augmentation"]["arguments"])
    if ('data_post' in execution):
        data_post = sak.class_selector(execution["data_post"]["class"])(**execution["data_post"]["arguments"])

    # Select iterator decorator
    train_type = 'Train' if model.training else 'Valid'
    iterator = sak.get_tqdm(dataloader, execution['iterator'], 
                            desc="({}) Epoch {:>3d}/{:>3d}, Loss {:0.3f}".format(train_type, 
                                                                                 state['epoch']+1, 
                                                                                 execution['epochs'], np.inf))

    # Iterate over all data in train/validation/test dataloader:
    print_loss = np.inf
    for i, inputs in enumerate(iterator):
        # Apply data transforms
        if ('data_pre' in execution):
            data_pre(inputs=inputs)
        if ('augmentation' in execution) and model.training:
            augmentation(inputs=inputs)
        if ('data_post' in execution):
            data_post(inputs=inputs)

        # Map all inputs to device
        for k in inputs:
            inputs[k] = inputs[k].to(state['device'], non_blocking=True)

        # Set gradient to zero
        if model.training: 
            state['optimizer'].zero_grad()

        # Predict input data
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(inputs=inputs,outputs=outputs)

        # Break early
        if torch.isnan(loss):
            raise ValueError("Nan loss value encountered. Stopping...")

        # Retrieve for printing purposes
        print_loss = loss.item()
        
        # Optimize network's weights
        if model.training:
            loss.backward()
            state['optimizer'].step()

        # Accumulate losses
        batch_loss[i] = print_loss

        # Change iterator description
        if isinstance(iterator,tqdm.tqdm):
            if i == len(iterator)-1:
                iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}".format(train_type, state['epoch']+1, execution['epochs'], np.mean(batch_loss)))
            else:
                iterator.set_description("({}) Epoch {:>3d}/{:>3d}, Loss {:10.3f}".format(train_type, state['epoch']+1, execution['epochs'], print_loss))
            iterator.refresh()

    return batch_loss


def train_model(model, state: dict, execution: dict, loader: torch.utils.data.DataLoader):
    # Send model to device
    model = model.to(state['device'])

    # Instantiate criterion
    criterion = sak.class_selector(execution['loss']['class'])(**execution['loss'].get('arguments',{}))
    
    # Initialize best loss for early stopping
    if 'best_loss' not in state:
        state['best_loss'] = np.inf

    for epoch in range(state['epoch'], execution['epochs']):
        try:
            # Store current epoch
            state['epoch'] = epoch
            
            # Train model
            loss_train = do_epoch(model.train(), state, execution, loader, criterion)
            state['loss_train'] = np.mean(loss_train)

            # Save model/state info
            torch.save(model, os.path.join(execution['save_directory'],'checkpoint.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'checkpoint.state'), mode='wb')
            
            # Check if loss is best loss
            if state['loss_train'] < state['best_loss']:
                state['best_loss'] = state['loss_train']
                state['best_epoch'] = epoch
                
                # Copy checkpoint and mark as best
                shutil.copyfile(os.path.join(execution['save_directory'],'checkpoint.model'), os.path.join(execution['save_directory'],'model_best.model'))
                shutil.copyfile(os.path.join(execution['save_directory'],'checkpoint.state'), os.path.join(execution['save_directory'],'model_best.state'))
        except KeyboardInterrupt:
            torch.save(model, os.path.join(execution['save_directory'],'keyboard_interrupt.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'keyboard_interrupt.state'), mode='wb')
            raise
        except:
            torch.save(model, os.path.join(execution['save_directory'],'error.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'error.state'), mode='wb')
            raise


def train_valid_model(model, state: dict, execution: dict, 
                      loader_train: torch.utils.data.DataLoader, 
                      loader_valid: torch.utils.data.DataLoader):
    # Send model to device
    model = model.to(state['device'])

    # Instantiate criterion
    criterion = sak.class_selector(execution['loss']['class'])(**execution['loss'].get('arguments',{}))
    
    # Initialize best loss for early stopping
    if 'best_loss' not in state:
        state['best_loss'] = np.inf

    for epoch in range(state['epoch'], execution['epochs']):
        try:
            # Store current epoch
            state['epoch'] = epoch
            
            # Training model
            loss_train = do_epoch(model.train(), state, execution, loader_train, criterion)
            state['loss_train'] = np.mean(loss_train)

            # Validate results
            with torch.no_grad():
                loss_valid = do_epoch(model.eval(), state, execution, loader_valid, criterion)
            state['loss_validation'] = np.mean(loss_valid)

            # Update learning rate scheduler
            if 'scheduler' in state:
                state['scheduler'].step(state['loss_validation'])

            # Save model/state info
            torch.save(model, os.path.join(execution['save_directory'],'checkpoint.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'checkpoint.state'), mode='wb')
            
            # Check if loss is best loss
            compound_loss = 2*state['loss_train']*state['loss_validation']/(state['loss_train']+state['loss_validation'])
            if compound_loss < state['best_loss']:
                state['best_loss'] = compound_loss
                state['best_epoch'] = epoch
                
                # Copy checkpoint and mark as best
                shutil.copyfile(os.path.join(execution['save_directory'],'checkpoint.model'), os.path.join(execution['save_directory'],'model_best.model'))
                shutil.copyfile(os.path.join(execution['save_directory'],'checkpoint.state'), os.path.join(execution['save_directory'],'model_best.state'))
            
        except KeyboardInterrupt:
            torch.save(model, os.path.join(execution['save_directory'],'keyboard_interrupt.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'keyboard_interrupt.state'), mode='wb')
            raise
        except:
            torch.save(model, os.path.join(execution['save_directory'],'error.model'), pickle_module=dill)
            sak.pickledump(state, os.path.join(execution['save_directory'],'error.state'), mode='wb')
            raise
