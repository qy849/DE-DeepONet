import torch
from abc import ABC, abstractmethod 

class EarlyStoppingCriterion(ABC): 
    @abstractmethod
    def decide(self, train_losses_history, valid_losses_history): 
        """
        Arguments example:
        train_losses_history = {
            'eval': [eval_loss_1, eval_loss_2, ...],
            'dm': [dm_loss_1, dm_loss_2, ...],
            'dx': [dx_loss_1, dx_loss_2, ...],
            'total': [total_loss_1, total_loss_2, ...]
        }
        valid_losses_history = {
            'eval': [eval_loss_1, eval_loss_2, ...],
            'dm': [dm_loss_1, dm_loss_2, ...],
            'dx': [dx_loss_1, dx_loss_2, ...],
            'total': [total_loss_1, total_loss_2, ...]
        }
        """
        pass

class NoStop(EarlyStoppingCriterion):
    def __init__(self):
        self.stop = False
    
    def decide(self, train_losses_history, valid_losses_history):
        pass

class StopAtPatience(EarlyStoppingCriterion):
    def __init__(self, patience):
        self.patience = patience
        
        self.counter = 0
        self.stop = False
        self.min_valid_loss = torch.inf
    
    def decide(self, train_losses_history, valid_losses_history):
        if valid_losses_history['total']:
            most_recent_valid_loss = valid_losses_history['total'][-1]
            if most_recent_valid_loss < self.min_valid_loss:
                self.min_valid_loss = most_recent_valid_loss
                self.counter = 0
            else:
                self.counter += 1
        if self.counter > self.patience:
            self.stop = True