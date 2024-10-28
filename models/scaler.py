import torch
import torch.nn as nn


class StandardScaler(nn.Module):
    def __init__(self, x, eps=1e-10):
        super().__init__()
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, keepdim=True)
        self.eps = torch.tensor(eps, dtype=x.dtype)
        assert self.eps >= 0
        
    def transform(self,x):
        return (x-self.mean)/(self.std+self.eps)

    def inverse_transform(self,x):
        return x*(self.std+self.eps)+self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)
        return self
    

class StandardScaler_v2(nn.Module):
    def __init__(self, x, eps=1e-10):
        super().__init__()
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = torch.tensor(eps, dtype=x.dtype)
        assert self.eps >= 0
        
    def transform(self,x):
        return (x-self.mean)/(self.std+self.eps)

    def inverse_transform(self,x):
        return x*(self.std+self.eps)+self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)
        return self

    
class MinMaxScaler(nn.Module):
    def __init__(self, x, bound=1e-2):
        super().__init__()
        self.min = torch.min(x,dim=0)[0]
        self.max = torch.max(x,dim=0)[0]
        self.bound = torch.tensor(bound, dtype=x.dtype)
        assert self.bound > 0
        
    def transform(self,x):
        return 2*self.bound*(x-self.min)/(self.max-self.min)-self.bound

    def inverse_transform(self,x):
        return (self.max-self.min)*(x+self.bound)/(2*self.bound)+self.min
    
    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        self.bound = self.bound.to(device)
        return self
    
    
class IdentityScaler(nn.Module):
    def __init__(self):
        super().__init__()
        
    def transform(self,x):
        return x

    def inverse_transform(self,x):
        return x
    
    def to(self, device):
        self = super().to(device)
        return self
