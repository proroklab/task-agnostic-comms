import torch
from typing import Optional

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None) -> torch.Tensor:
	index = broadcast(index, src, dim)
	size = list(src.size())
	if dim_size is not None:
		size[dim] = dim_size
	elif index.numel() == 0:
		size[dim] = 0
	else:
		size[dim] = int(index.max()) + 1
	out = torch.zeros(size, dtype=src.dtype, device=src.device)
	return out.scatter_add_(dim, index.long(), src)

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
	if dim < 0:
		dim = other.dim() + dim
	if src.dim() == 1:
		for _ in range(0, dim):
			src = src.unsqueeze(0)
	for _ in range(src.dim(), other.dim()):
		src = src.unsqueeze(-1)
	src = src.expand_as(other)
	return src

def size_nested(input, dim):
	if dim == 0:
		size = torch.tensor(len(input._nested_tensor_size()))
	else:
		dim = dim - 1 if dim > 0 else dim
		size = torch.tensor([size[dim] for size in input._nested_tensor_size()])
		if torch.all(size==size[0]):
			size = size[0]
	return size

def shape_nested(input):
	if input.is_nested:
		shape = []
		num_dims = input.dim()
		for dim in range(num_dims):
			dim_size = size_nested(input, dim=dim)
			if len(dim_size.shape) > 0:
				dim_size = None
			else:
				dim_size = dim_size.item()
			shape.append(dim_size)
		return tuple(shape)
	else:
		return tuple(input.shape)

def nested_to_batch(nested, return_sizes=False):
	tensor_list = nested.unbind()
	flat = torch.cat(tensor_list, dim=0)
	sizes = size_nested(nested, dim=1)
	if return_sizes:
		return flat, sizes
	else:
		batch = torch.arange(len(tensor_list)).repeat_interleave(sizes)
		return flat, batch

def index_with_nested(src, index, dim=0):
	return torch.nested_tensor([select_index(src=src, idx=idxi.long(), dim=dim) for idxi in index.unbind()])

def permute_nested(nt, perm):
	if perm.is_nested:
		return torch.nested_tensor([xi[permi] for xi, permi in zip(nt.unbind(), perm.unbind())])
	else:
		dim0_size = size_nested(nt, dim=0)
		x_flat, sizes = nested_to_batch(nt, return_sizes=True)
		batch = torch.arange(dim0_size).repeat_interleave(sizes)
		x_perm = x_flat[perm]
		batch = batch[perm]
		return create_nested_batch(x_perm, batch, dim_size=dim0_size)

def create_nested(x, sizes):
	return torch.nested_tensor(torch.split(x, sizes.tolist()))

def create_nested_batch(x, batch, dim_size=None):
	if dim_size is None:
		dim_size = torch.max(batch)+1
	sizes = torch.zeros(dim_size).scatter_(dim=0, index=batch, src=torch.ones(batch.shape[0]), reduce='add').int()
	return create_nested(x, sizes)

def truncate_nested(nt, sizes, dim=1):
	dim = dim-1
	size_update = lambda size, dimsize: tuple(list(size)[:dim] + [dimsize] + list(size)[dim+1:])
	return torch.nested_tensor([
			torch.cat([
				torch.narrow(xi, dim=dim, start=0, length=min(size, xi.shape[dim])),
				torch.full(size=size_update(xi.shape, max(size-xi.shape[dim], 0)), fill_value=torch.nan)
			], dim=dim)
		for (xi, size) in zip(nt.unbind(), sizes)])


def combine_dicts(dict_list):
	keys = dict_list[0].keys()
	return {
		key:
			sum([dict_list[i][key] for i in range(len(dict_list))]) / len(dict_list)
		for key in keys
	}


def mul_nested(input1, input2):
	return torch.nested_tensor([x1i * x2i for (x1i, x2i) in zip(input1.unbind(), input2.unbind())])


def add_nested(input1, input2):
	return torch.nested_tensor([x1i * x2i for (x1i, x2i) in zip(input1.unbind(), input2.unbind())])


def cat_nested(input1, input2, dim):
	'''
	:param input1: First element to concatenate (tensor or nested_tensor)
	:param input2: Second element to concatenate (tensor or nested_tensor)
	:param dim: Dimension along which to concatenate
	:return: The concatenated nested_tensor

	input1 and input2 must have the same number of dimensions, and the sizes in all dimensions except dim must either match
	or be 1. If the dimension in one of the tensors is 1, then it is broadcast to the shape of the other tensor in that dimension.
	This functionality is important because it is non-trivial to broadcast manually with nested_tensors. For example, if
	we wish to concatenate input1: (B x 1 x D1) to input2: (B x None x D2) in dimension -1, producing an output: (B x None x (D1+D2)),
	then it would be nice if input1 were automatically broadcast to each element in the None dimension so that we do not first
	need to construct a nested_tensor.
	'''
	if dim == 0:
		return torch.nested_tensor(input1.unbind() + input2.unbind())
	dim = dim - 1 if dim > 0 else dim
	if (input1.is_nested) and (not input2.is_nested):
		dim0_size = len(input1.unbind())
		input2 = input2.expand(dim0_size, *input2.shape[1:])
	if (input2.is_nested) and (not input1.is_nested):
		dim0_size = len(input2.unbind())
		input1 = input1.expand(dim0_size, *input1.shape[1:])
	out = []
	for x1i, x2i in zip(input1.unbind(), input2.unbind()):
		x1i_shape = torch.tensor(x1i.shape)
		x2i_shape = torch.tensor(x2i.shape)
		if torch.any(x1i_shape == 0) or torch.any(x2i_shape == 0):
			out_shape = torch.maximum(x1i_shape, x2i_shape)
			out_shape[dim] = x1i_shape[dim] + x2i_shape[dim]
			out_shape[x1i_shape == 0] = 0
			out_shape[x2i_shape == 0] = 0
			outi = torch.zeros(*out_shape)
			out.append(outi)
		else:
			x1i_shape[dim] = -1
			x2i_shape[dim] = -1
			out_shape = torch.maximum(x1i_shape, x2i_shape)
			outi = torch.cat([
				x1i.expand(*out_shape),
				x2i.expand(*out_shape),
			], dim=dim)
			out.append(outi)
	return torch.nested_tensor(out)

def sum_nested(input, dim):
	dim = dim - 1 if dim > 0 else dim
	x = [xi.sum(dim=dim) for xi in input.unbind()]
	if all(map(lambda xi: xi.shape == x[0].shape, x)):
		return torch.stack(x, dim=0)
	else:
		return torch.nested_tensor(x)


def select_index(src, idx, dim):
	idx_list = [slice(None)] * len(src.shape)
	idx_list[dim] = idx
	return src.__getitem__(idx_list)