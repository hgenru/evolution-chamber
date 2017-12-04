import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import random

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

	def make_child(self, mutate=False):
		params = []
		for name in self.state_dict():
			if name.split('.')[-1] != 'bias':
				params.append(list(self.state_dict()[name].size()))

		if mutate:
			for layer in params:
				for i, n in enumerate(layer):
					layer[i] = int(n + math.floor( random.randint(0, n//5) - n//10 ))

		return NetModel(layer_sizes=params)


# model = NetModel([
# 	[100, 1],
# 	[100, 1],
# 	[100, 1],
# 	[100, 1],
# ])
# child = model.make_child()
# mutated_child = model.make_child(mutate=True)
