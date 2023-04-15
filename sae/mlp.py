import torch.nn as nn
from torch import Tensor
import torch
import numpy as np
from torch.nn import MSELoss



def build_mlp(input_dim, output_dim, nlayers=1, midmult=1., batchnorm=False, layernorm=True, nonlinearity=nn.GELU):
	mlp_layers = layergen(input_dim=input_dim, output_dim=output_dim, nlayers=nlayers, midmult=midmult)
	mlp = MLP(layer_sizes=mlp_layers, batchnorm=batchnorm, layernorm=layernorm, nonlinearity=nonlinearity)
	return mlp



def layergen(input_dim, output_dim, nlayers=1, midmult=1.0):
	midlayersize = midmult * (input_dim + output_dim) // 2
	midlayersize = max(midlayersize, 1)
	nlayers += 2
	layers1 = np.around(
		np.logspace(np.log10(input_dim), np.log10(midlayersize), num=(nlayers) // 2)
	).astype(int)
	layers2 = np.around(
		np.logspace(
			np.log10(midlayersize), np.log10(output_dim), num=(nlayers + 1) // 2
		)
	).astype(int)[1:]
	return list(np.concatenate([layers1, layers2]))



class MLP(nn.Module):
	def __init__(
		self,
		layer_sizes,
		batchnorm=False,
		layernorm=False,
		nonlinearity=nn.ReLU,
	):
		super(MLP, self).__init__()
		self.batchnorm = batchnorm
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i != len(layer_sizes) - 2:
				if batchnorm:
					layers.append(BatchNorm(layer_sizes[i + 1]))
				if layernorm:
					layers.append(nn.LayerNorm(layer_sizes[i + 1]))
				layers.append(nonlinearity())
		self.net = nn.Sequential(*layers)

	def forward(self, X):
		return self.net(X)



class MLPForwardReverse(nn.Module):

	def __init__(
		self,
		layer_sizes=[1, 2, 4, 2, 1],
		monotonic=False,
		activation=nn.Tanh,
	):
		super().__init__()
		self.layer_sizes = layer_sizes
		self.monotonic = monotonic
		self.activation = activation
		layers_forward = []
		layers_reverse = []
		for i in range(len(self.layer_sizes) - 1):
			if self.monotonic:
				layers_forward.append(LinearAbs(layer_sizes[i], layer_sizes[i + 1]))
				layers_reverse.append(LinearAbs(layer_sizes[i], layer_sizes[i + 1]))
			else:
				layers_forward.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
				layers_reverse.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i != len(layer_sizes) - 2:
				layers_forward.append(self.activation())
				layers_reverse.append(self.activation())
		self.net_for = nn.Sequential(*layers_forward)
		self.net_rev = nn.Sequential(*layers_reverse)
		self.mse = MSELoss()
		self.inverse_loss = 0
		self.register_forward_hook(self.inverse_objective)

	def inverse_objective(self, module, grad_input, grad_output):
		x = self.input_forward
		xhat = self.net_rev(self.net_for(x))
		self.inverse_loss = self.mse(x, xhat)
		self.inverse_loss.backward()

	def forward(self, input: Tensor) -> Tensor:
		shape = input.shape
		x = input.reshape(-1, 1)
		y = self.net_for(x)
		self.input_forward = x
		return y.reshape(shape)

	def reverse(self, input: Tensor) -> Tensor:
		shape = input.shape
		x = input.reshape(-1, 1)
		y = self.net_rev(x)
		return y.reshape(shape)



class Elementwise(nn.Module):

	def __init__(self, dim, params=None, bias=True):
		super().__init__()
		self.features = dim
		self.include_bias = bias
		self.num_params = dim+1 if self.include_bias else dim
		self.batch = None
		self.weight = None
		self.bias = None
		if params is not None:
			self.batch = params.shape[0]
			self.set_params(params)

	def __repr__(self):
		return "Elementwise(features={feat}, batch={batch}, bias={bias})".format(feat=self.features, batch=self.batch, bias=self.include_bias)

	def set_params(self, params):
		self.batch = params.shape[0]
		self.weight = params[:,:self.features].reshape(self.batch, self.features)
		if self.include_bias:
			self.bias = params[:,self.features:self.num_params].reshape(self.batch, 1)

	def forward(self, x):
		if len(x.shape) == 2:
			# x: batch x in_dim
			y = x * self.weight
			if self.include_bias:
				y += self.bias
			return y
		elif len(x.shape) == 3:
			# x: batch x N x in_dim
			y = x.unsqueeze(1) * self.weight
			if self.include_bias:
				y += self.bias
			return y
		else:
			raise ValueError("Input shape of {shape} not valid in LinearHyper".format(x.shape))



class MultiModule(nn.Module):

	def __init__(self, n_agents, module=MLP, *args, **kwargs):
		super().__init__()
		self.n_agents = n_agents
		self.mlps = nn.ModuleList([module(*args, **kwargs) for _ in range(self.n_agents)])
		self.mlp_modules = list(self.mlps.children())

	def forward(self, X):
		# B1, ..., Bn, n_agents, input_dim
		return torch.stack([self.mlp_modules[i](select_index(X, -2, i)) for i in range(self.n_agents)], dim=-2)



class LinearAbs(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, input: Tensor) -> Tensor:
		return nn.functional.linear(input, torch.abs(self.weight), self.bias)


class BatchNorm(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.bn = nn.BatchNorm1d(*args, **kwargs)

	def forward(self, x):
		shape = x.shape
		x_r = x.reshape(np.prod(shape[:-1]), shape[-1])
		y_r = self.bn(x_r)
		y = y_r.reshape(shape)
		return y



def select_index(arr, dim, idx):
	idx_list = [slice(None)] * len(arr.shape)
	idx_list[dim] = idx
	return arr.__getitem__(idx_list)



if __name__ == '__main__':
	input_dim = 3
	output_dim = 5
	f = MultiModule(8, MLP, layer_sizes=[input_dim, output_dim])
	x = torch.randn(10, 8, 3)
	y = f(x)
	print(y.shape)




## Unused

'''
class LinearInv(nn.Linear):
	def __init__(self, *args, monotonic=False, **kwargs):
		super().__init__(*args, **kwargs)

		self.monotonic = monotonic
		self.weight_inv = None

		def backward_hook(module, *args, **kwargs):
			module.weight_inv = None

		self.register_full_backward_hook(backward_hook)

	def forward(self, input: Tensor) -> Tensor:
		# Monotonic
		if self.monotonic:
			weight = torch.abs(self.weight)
		else:
			weight = self.weight
		# Compute W^-1
		if self.weight_inv == None:
			if weight.shape[0] == weight.shape[1]:
				self.weight_inv = torch.inverse(weight)
			else:
				self.weight_inv = torch.pinverse(weight.detach())
		# Linear
		return nn.functional.linear(input, weight, self.bias)

	def reverse(self, input: Tensor) -> Tensor:
		y = input - self.bias
		x = nn.functional.linear(input=y, weight=self.weight_inv)
		return x
	
	
class ILU(nn.Module):
	def __init__(self, alpha=1.0, learnable=True):
		super().__init__()
		self.alpha = (
			torch.nn.Parameter(torch.tensor(alpha))
			if learnable
			else torch.tensor(alpha)
		)

	def forward(self, input: Tensor) -> Tensor:
		mask = input < 0
		y = input.clone()
		y[mask] = -1 / self.alpha * torch.log(
			1 / self.alpha - input[mask]
		) + 1 / self.alpha * torch.log(1 / self.alpha)
		return y

	def reverse(self, input: Tensor) -> Tensor:
		mask = input < 0
		y = input.clone()
		y[mask] = 1 / self.alpha - 1 / self.alpha * torch.exp(-self.alpha * input[mask])
		return y
'''
