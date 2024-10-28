import torch.nn as nn

class ScaledModel(nn.Module):

    def __init__(self, scaler, model):
        super().__init__()
        self.scaler = scaler
        self.model = model

    def to(self, device):
        self.scaler = self.scaler.to(device)
        self.model = self.model.to(device)
        return self

    def train(self, mode=True):
        super().train(mode)  
        self.model.train(mode) 

    def eval(self):
        super().eval()  
        self.model.eval() 

    def forward(self, branch_inputs, trunk_inputs):
        return self.model(self.scaler.transform(branch_inputs), trunk_inputs)

class ScaledModel_v2(nn.Module):

    def __init__(self, scaler, model):
        super().__init__()
        self.scaler = scaler
        self.model = model

    def to(self, device):
        self.scaler = self.scaler.to(device)
        self.model = self.model.to(device)
        return self

    def train(self, mode=True):
        super().train(mode)  
        self.model.train(mode) 

    def eval(self):
        super().eval()  
        self.model.eval() 

    def forward(self, branch_inputs):
        return self.model(self.scaler.transform(branch_inputs))