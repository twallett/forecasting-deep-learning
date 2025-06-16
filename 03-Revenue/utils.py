import os
import torch
import random
import numpy as np

class EarlyStopping:
    def __init__(self, model,patience=5, delta=0, verbose=False):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = 0

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
            
        if self.verbose:
            print(f'Validation score improved {self.val_score:.4f} --> {val_score:.4f} - Saving model...')
            
        if not os.path.exists('save'):
            os.makedirs('save')
            
        torch.save(model, os.path.join('save', f'{self.model}.pt'))
        
        self.val_score = val_score
        
        
def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True