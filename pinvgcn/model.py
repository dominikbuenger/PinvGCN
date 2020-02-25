
import torch

import torch.nn.functional as F
import torch_geometric.nn.inits as inits

    

class PinvConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, is_last=False):
        super().__init__()
        self.is_last = is_last

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.weight)
        inits.zeros(self.bias)

    def forward(self, x, data):
        x = data.apply_pinv(x, self.weight, training = self.training and self.is_last)
        if self.bias is not None:
            x = x + self.bias
        return x

class GlorotLinear(torch.nn.Linear):
    def reset_parameters(self):
        inits.glorot(self.weight)
        inits.zeros(self.bias)


class PinvGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=[32], dropout=0.5, bias=True):
        super().__init__()
        self.depth = len(hidden)+1
        self.dropout = None if dropout is None or dropout == 0 else torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(GlorotLinear(in_channels, hidden[0], bias=bias))
        for i in range(self.depth-2):
            self.layers.append(PinvConv(hidden[i], hidden[i+1], bias=bias))
        self.layers.append(PinvConv(hidden[-1], out_channels, bias=bias, is_last=True))
        
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def reg_params(self):
        return [p for layer in self.layers[:-1] for p in layer.parameters()]

    def non_reg_params(self):
        return self.layers[-1].parameters()
            
    def forward(self, data):
        X = data.pinv_preconvolved_x
        if X is None:
            raise "data.prepare_training must be called before training"
        X = self.layers[0](X)
        
        for i in range(1, self.depth):
            X = self.activation(X)
            if self.dropout is not None:
                X = self.dropout(X)
            X = self.layers[i](X, data)
            
        return X
    
    def run_training(self, data, optimizer, num_epochs):
        self.train()
        data.prepare_training()
        y = data.y[data.train_mask]
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self(data)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def eval_accuracy(self, data):
        self.eval()
        _, pred = self(data)[data.test_mask].max(dim=1)
        return float(pred.eq(data.y[data.test_mask]).sum().item()) / data.test_mask.sum().item()
