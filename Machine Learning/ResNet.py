import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, inp_dim, hid_dim, act_fn) -> None:
        super().__init__()
        self.lin1=nn.Linear(inp_dim, hid_dim)
        self.bn1=nn.BatchNorm1d(hid_dim)
        self.act1=act_fn()
        
        self.lin2=nn.Linear(hid_dim, inp_dim)
        self.bn2=nn.BatchNorm1d(inp_dim)
        self.act2=act_fn()
        
    def forward(self, inp):
        temp_inp=self.lin1(inp)
        temp_inp=self.bn1(temp_inp)
        temp_inp=self.act1(temp_inp)
        
        temp_inp=self.lin2(temp_inp)
        temp_inp=self.bn2(temp_inp)
        out=self.act2(temp_inp+inp)
        return out