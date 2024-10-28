import torch
import torch.nn as nn

# see https://pytorch.org/docs/stable/nn.html#loss-functions
def get_loss_function(name):

    if name in ['l1_loss', 'L1']:
        return nn.L1Loss
    elif name in ['mse_loss', 'MSELoss']:
        return nn.MSELoss
    elif name in ['huber_loss', 'HuberLoss']:
        return nn.HuberLoss
    elif name in ['smooth_l1_Loss', 'SmoothL1Loss']:
        return nn.SmoothL1Loss
    else:
        raise NotImplementedError

class WeightedMSELoss(nn.Module):
    def __init__(self, weighted_matrix, reduction='sum'):
        super().__init__()
        self.weighted_matrix = weighted_matrix
        self.reduction = reduction

        if self.reduction not in ['sum', 'mean']:
            raise NotImplementedError

    def to(self, device):
        self.weighted_matrix = self.weighted_matrix.to(device)
        return self

    def forward(self, outputs, labels):
        diff = outputs - labels
        loss =  torch.einsum('ij, jk, ik -> i', diff, self.weighted_matrix, diff)
        if self.reduction == 'sum':
            return torch.sum(loss, dim=0)
        else:
            return torch.mean(loss, dim=0)

class RelativeL2Loss(nn.Module):
    def __init__(self, eps=1e-10, reduction='sum'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def l2_norm(self, x):
        if x.dim() == 2:
            return torch.sum(x**2, dim=1)**(1/2)
        elif x.dim() == 3:
            return torch.sum(x**2, dim=(1,2))**(1/2)
        elif x.dim() == 4:
            return torch.sum(x**2, dim=(1,2,3))**(1/2)
        else:
            raise ValueError('Unsupported tensor shape')

    def forward(self, outputs, labels):
        diff = outputs - labels
        loss = self.l2_norm(diff) / (self.l2_norm(labels) + self.eps)
        if self.reduction == 'sum':
            return torch.sum(loss, dim=0)
        else:
            return torch.mean(loss, dim=0)


class RelativeWeightedL2Loss(nn.Module):
    def __init__(self, weighted_matrix, eps=1e-10, reduction='sum'):
        super().__init__()
        self.weighted_matrix = weighted_matrix
        self.eps = eps
        self.reduction = reduction

        if self.reduction not in ['sum', 'mean']:
            raise NotImplementedError

    def to(self, device):
        self.weighted_matrix = self.weighted_matrix.to(device)
        return self

    def weighted_l2_norm(self, x):
        if x.dim() == 2:
            return torch.einsum('ij, jk, ik -> i', x, self.weighted_matrix, x)**(1/2)
        else:
            return ValueError('Unsupported tensor shape')

    def forward(self, outputs, labels):
        diff = outputs - labels
        loss =  self.weighted_l2_norm(diff) / (self.weighted_l2_norm(labels) + self.eps)
        if self.reduction == 'sum':
            return torch.sum(loss, dim=0)
        else:
            return torch.mean(loss, dim=0)


class RelativeWeightedH1Loss(nn.Module):
    def __init__(self, weighted_matrix, eps=1e-10, reduction='sum'):
        super().__init__()
        self.weighted_matrix = weighted_matrix
        self.eps = eps
        self.reduction = reduction

        self.identity_matrix = torch.eye(weighted_matrix.shape[0])
        if self.reduction not in ['sum', 'mean']:
            raise NotImplementedError

    def to(self, device):
        self.weighted_matrix = self.weighted_matrix.to(device)
        self.identity_matrix = self.identity_matrix.to(device)
        return self

    def weighted_h1_norm(self, x):
        if x.dim() == 2:
            return torch.einsum('ij, jk, ik -> i', x, self.identity_matrix + self.weighted_matrix, x)**(1/2)
        else:
            return ValueError('Unsupported tensor shape')

    def forward(self, outputs, labels):
        diff = outputs - labels
        loss =  self.weighted_h1_norm(diff) / (self.weighted_h1_norm(labels) + self.eps)
        if self.reduction == 'sum':
            return torch.sum(loss, dim=0)
        else:
            return torch.mean(loss, dim=0)