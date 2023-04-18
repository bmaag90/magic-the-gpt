import torch
import torch.nn as nn
from datetime import datetime
import os
import yaml

def get_map_char_to_int(chars):
    ''' Creates a mapping from characters to indices
    '''
    return { ch:i for i,ch in enumerate(chars) }

def get_map_int_to_char(chars):
    ''' Creates a mapping from indices to character
    '''
    return { i:ch for i,ch in enumerate(chars) }

def encode(map_char_to_int, s):
    ''' Encode string to list of indices
    '''
    return [map_char_to_int[c] for c in s]

def decode(map_int_to_char, l):
    ''' Decode list of indices to string
    '''
    return ''.join([map_int_to_char[i] for i in l])

@torch.no_grad()
def estimate_loss(model, device, data_loader, eval_iters=100):
    ''' Estimate loss
    '''
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, (x, y) in enumerate(data_loader):
        if k == eval_iters:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        losses[k] = loss.item()
        
    model.train()
    return losses.mean()

def save_model_and_config(model, config):
    ''' Save model to .pth file and config to .yaml
    Saves both files to timestamped directory
    '''
    str_now = datetime.now().strftime("%y%m%d%H%M")
    save_path = os.path.join('./models', str_now)
    os.makedirs(save_path)

    torch.save(model, os.path.join(save_path, 'model.pth'))

    with open(os.path.join(save_path, 'config.yaml'), 'w') as fh:
        yaml.dump(config, fh)
