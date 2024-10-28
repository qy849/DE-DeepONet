import torch
from abc import ABC, abstractmethod 
  
class LossBalancingAlgorithm(ABC): 
  
    @abstractmethod
    def rebalance(self, losses, loss_weights):
        """
        Modifify loss_weithg

        Arguments example:
        losses = {
            'eval': nn.MSELoss(output_eval, label_eval),
            'dm': nn.MSELoss(output_dm, label_dm),
            'dx': nn.MSELoss(output_dx, label_dx)
        }
        loss_weights = {
            'eval': 1.0,
            'dm': 1.0,
            'dx': 1.0
        }
        """
        pass

class NoUpdate(LossBalancingAlgorithm):
    def __init__(self):
        self.counter = 0
    
    def rebalance(self, losses, loss_weights):
        self.counter += 1

class LearningRateAnnealing(LossBalancingAlgorithm):
    def __init__(self, model, optimizer, update_frequency: int=100, alpha: float=0.9):
        self.model = model
        self.optimizer = optimizer
        self.update_frequency = update_frequency
        self.alpha = alpha

        self.counter = 0

    def rebalance(self, losses, loss_weights):
        self.counter += 1
        keys = [key for key, value in losses.items() if value is not None]
        if len(keys) > 1:
            if self.counter % self.update_frequency == 0:
                self.optimizer.zero_grad()
                previous_grad = None
                l2_norm_grad = {key: 0.0 for key in keys}
                for key in keys:
                    individual_loss = loss_weights[key] * losses[key]
                    individual_loss.backward(retain_graph=True)
                    if previous_grad is None:
                        previous_grad = []
                        for param in self.model.parameters():
                            if param.requires_grad:
                                if param.grad is None:  
                                    param.grad = torch.zeros_like(param)
                                    previous_grad.append(param.grad.clone())
                                else:
                                    previous_grad.append(param.grad.clone())
                                l2_norm_grad[key] += torch.sum(previous_grad[-1] ** 2)
                    else:
                        for i, param in enumerate(self.model.parameters()):
                            if param.requires_grad:
                                previous_grad[i] = param.grad - previous_grad[i]
                                l2_norm_grad[key] += torch.sum(previous_grad[i] ** 2)
                    
                    l2_norm_grad[key] = torch.sqrt(l2_norm_grad[key])

                sum_l2_norm_grad = sum(l2_norm_grad.values())
                for key in keys:
                    loss_weights[key] = (self.alpha * loss_weights[key] + (1 - self.alpha) * sum_l2_norm_grad / l2_norm_grad[key]).item()