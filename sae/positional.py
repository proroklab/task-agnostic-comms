import torch
from torch import nn
from torch import Tensor
import math

class PositionalEncoding(nn.Module):

	def __init__(self, dim: int, mode: str = 'onehot'):
		super().__init__()
		self.dim = dim
		self.mode = mode
		self.I = torch.eye(self.dim).byte()

	def forward(self, x: Tensor) -> Tensor:
		if self.mode == 'onehot':
			return self.onehot(x.int()).float()
		elif self.mode == 'binary':
			return self.binary(x.int()).float()
		elif self.mode == 'sinusoid':
			return self.sinusoid(x)

	def onehot(self, x: Tensor) -> Tensor:
		out_shape = list(x.shape) + [self.dim]
		self.I = self.I.to(x.device)
		return torch.index_select(input=self.I, dim=0, index=x.reshape(-1)).reshape(*out_shape)

	def binary(self, x: Tensor) -> Tensor:
		x = x + 1
		mask = 2 ** torch.arange(self.dim).to(x.device, x.dtype)
		return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

	def binary_to_int(self, x: Tensor) -> Tensor:
		multiplier = 2 ** torch.arange(x.shape[-1]).float().view(-1,1)
		y = x.float() @ multiplier
		return (y-1).squeeze(1).int()

	def binary_logits_to_binary(self, x: Tensor) -> Tensor:
		xs = torch.softmax(x, dim=1)
		max_mag = torch.max(xs, dim=1)[0]
		xs_reg = xs / max_mag[:,None]
		binary = (xs_reg > 0.5).int()
		return binary

	def onehot_logits_to_int(self, x: Tensor) -> Tensor:
		return torch.argmax(x, dim=-1)

	def sinusoid(self, x: Tensor):
		max_n = torch.max(x)+1
		pe = torch.zeros(max_n, self.dim, device=x.device)  # like 10x4
		position = torch.arange(0, max_n, dtype=torch.float, device=x.device).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.dim, 2, device=x.device).float() * (-math.log(10000.0) / self.dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe



if __name__ == '__main__':
	from torch.nn.functional import cross_entropy

	dim = 3
	max_n = 5
	batch_size = 16

	pos = PositionalEncoding(dim=3, mode='binary')
	x = torch.randn(10,3)
	b = pos.binary_logits_to_binary(x)
	i = pos.binary_to_int(b)

	pos = PositionalEncoding(dim=6, mode='binary')
	k = torch.arange(4)
	keys = pos(k)
	n = pos.binary_to_int(keys)
	y = torch.rand(k.shape[0], 6)
	loss1 = cross_entropy(y, k, reduction='none')
	loss2 = cross_entropy(y, keys, reduction='none')

	pos = PositionalEncoding(dim=8, mode='sinusoid')
	k = torch.arange(5)
	keys = pos(k)
