import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class SimpleLayer(nn.Module):
	def __init__(self, layer_size, activation=False):
		super(SimpleLayer, self).__init__()
		self.linear = nn.Linear(layer_size[1], layer_size[0])
		self.activation = activation
	def forward(self, x):
		return self.linear( x ) if not self.activation else self.activation( self.linear( x ) )


class NetModel(nn.Module):
    def __init__(self, layer_sizes):
        super(NetModel, self).__init__()
        self.layer_list = []
        for layer_size in layer_sizes:
            self.layer_list.append(SimpleLayer(layer_size))

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        return self.net( x )

    def get_score(self):
        pass

    def make_child(self, mutate=True):
        params = []
        for name in self.state_dict():
            if name.split('.')[-1] != 'bias':
                params.append(self.state_dict()[name].size())

        return NetModel(params)

# model = NetModel([
#     [10, 1]
# ])
# model(x)
# model.forward(x)
# child = model.make_child()
# print(child.state_dict())
# print([param in model.state_dict()])
