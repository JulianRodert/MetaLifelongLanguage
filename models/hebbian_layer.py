import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BaseHebbian(nn.Module):

    def __init__(self, in_features, out_features, activation=F.softmax):
        super().__init__()

        self.activation = activation

        # The matrix of fixed (baseline) weights
        self.w = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Just a bias term
        self.b = Parameter(.01 * torch.randn(1), requires_grad=True)
        # The matrix of plasticity coefficients
        self.alpha = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # The weight decay term - "learning rate" of plasticity - trainable, but shared across all connections
        self.eta = Parameter(.01 * torch.ones(1), requires_grad=True)
        # Initialize Hebb
        self.hebb = torch.zeros(in_features, out_features)

    def forward(self, x):
        x_in = None
        if self.training:
            x_in = x.detach().clone()

        x = self.activation(x.mm(self.w + self.alpha * self.hebb) + self.b)

        if self.training:
            x_out = x.detach().clone()
            with torch.no_grad():
                self.update_hebbian_trace(x_in, x_out)

        return x

    def update_hebbian_trace(self, x_in, x_out):
        hebb = (1 - self.eta) * self.hebb + self.eta * torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1))[0]
        self.hebb = torch.clamp(hebb, -1, 1)

    def reset_hebbian_trace(self):
        self.hebb *= 0


class SimpleHebbian(BaseHebbian):

    def __init__(self, in_features, out_features, activation=F.softmax):
        super().__init__(in_features, out_features, activation)

        self.eta = Parameter(.01 * torch.ones(in_features, out_features), requires_grad=True)


class OjaHebbian(SimpleHebbian):

    def update_hebbian_trace(self, x_in, x_out):
        hebb = self.hebb + self.eta * torch.mul((x_in[0].unsqueeze(1) - torch.mul(self.hebb, x_out[0].unsqueeze(0))),
                                                x_out[0].unsqueeze(0))
        self.hebb = torch.clamp(hebb, -1, 1)

