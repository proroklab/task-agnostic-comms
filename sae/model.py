import torch
from torch import nn
from sae.util import scatter
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation, mean_squared_loss


class AutoEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        '''
      Must have self.encoder and self.decoder objects, which follow the encoder and decoder interfaces
      '''
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def forward(self, x, batch=None):
        z = self.encoder(x, batch)
        xr, batchr = self.decoder(z)
        return xr, batchr

    def get_vars(self):
        self.vars = {
            "n_pred_logits": self.decoder.get_n_pred_logits(),
            "n_pred": self.decoder.get_n_pred(),
            "n": self.encoder.get_n(),
            "x": self.encoder.get_x(),
            "xr": self.decoder.get_x_pred(),
        }
        return self.vars

    def loss(self, vars=None):
        '''
      Input: the output of self.get_vars()
      Returns: a dict of info which must include 'loss'
      '''
        if vars is None:
            vars = self.get_vars()
        pred_idx, tgt_idx = get_loss_idxs(vars["n_pred"], vars["n"])
        x = vars["x"]
        xr = vars["xr"]
        mse_loss = torch.mean(mean_squared_loss(x[tgt_idx], xr[pred_idx]))
        size_loss = torch.mean(mean_squared_loss(vars["n_pred_logits"], vars["n"].unsqueeze(-1).detach().float()))
        if torch.isnan(mse_loss):
            mse_loss = 0
        loss = 100 * mse_loss + 1e-3 * size_loss
        corr = correlation(x[tgt_idx], xr[pred_idx])
        return {
            "loss": loss,
            "size_loss": size_loss,
            "mse_loss": mse_loss,
            "corr": corr,
        }


class Encoder(nn.Module):

    def __init__(self, dim, hidden_dim=64, max_n=64, **kwargs):
        super().__init__()
        # Params
        self.input_dim = dim
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.pos_mode = kwargs.get("pos_mode", "onehot")
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
        self.key_net = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1.,
                                 layernorm=True, nonlinearity=nn.Mish)
        self.val_net = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1.,
                                 layernorm=True, nonlinearity=nn.Mish)
        self.rank = torch.nn.Linear(self.input_dim, 1)
        self.cardinality = torch.nn.Linear(1, self.hidden_dim)

    def sort(self, x, batch):
        mag = self.rank(x)
        max_mag = torch.max(mag) + 0.0001
        batch_mag = batch * max_mag
        new_mag = mag.squeeze() + batch_mag
        _, idx_sorted = torch.sort(new_mag)
        x_sorted = x[idx_sorted]
        xs_idx = idx_sorted
        xs = x_sorted
        return xs, xs_idx

    def forward(self, x, batch=None, n_batches=None):
        # x: n x input_dim
        _, input_dim = x.shape
        if batch is None:
            batch = torch.zeros(x.shape[0], device=x.device)
        self.batch = batch

        n = scatter(src=torch.ones(x.shape[0], device=x.device), index=batch, dim_size=n_batches).long()  # batch_size
        self.n = n

        # Sort
        xs, xs_idx = self.sort(x, batch)
        self.xs = xs
        self.xs_idx = xs_idx

        keys = torch.cat([torch.arange(ni, device=x.device) for ni in n], dim=0).int()  # batch_size]
        pos = self.pos_gen(keys)  # batch_size x hidden_dim

        # Encoder
        y = self.val_net(xs) * self.key_net(pos)  # total_nodes x hidden_dim
        z_elements = scatter(src=y, index=batch, dim=-2, dim_size=n_batches)  # batch_size x dim
        n_enc = self.cardinality(n.unsqueeze(-1).float())
        z = z_elements + n_enc
        self.z = z
        return z

    def get_x_perm(self):
        'Returns: the permutation applied to the inputs (shape: ninputs)'
        return self.xs_idx

    def get_z(self):
        'Returns: the latent state (shape: batch x hidden_dim)'
        return self.z

    def get_batch(self):
        'Returns: the batch idxs of the inputs (shape: ninputs)'
        return self.batch

    def get_x(self):
        'Returns: the sorted inputs, x[x_perm] (shape: ninputs x dim)'
        return self.xs

    def get_n(self):
        'Returns: the number of elements per batch (shape: batch)'
        return self.n

    def get_max_n(self):
        return self.max_n


class Decoder(nn.Module):

    def __init__(self, dim, hidden_dim=64, max_n=64, **kwargs):
        super().__init__()
        # Params
        self.output_dim = dim
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.pos_mode = kwargs.get("pos_mode", "onehot")
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
        self.key_net = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1.,
                                 layernorm=True, nonlinearity=nn.Mish)
        self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=2, midmult=1.,
                                 layernorm=False, nonlinearity=nn.Mish)
        self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=1, nlayers=2, layernorm=True,
                                   nonlinearity=nn.Mish)

    def forward(self, z):
        # z: batch_size x hidden_dim
        n_logits = self.size_pred(z)  # batch_size x max_n
        n = torch.round(n_logits, decimals=0).squeeze(-1).int()
        n = torch.minimum(n, torch.tensor(self.max_n - 1))
        n = torch.maximum(n, torch.tensor(0))
        self.n_pred_logits = n_logits
        self.n_pred = n

        k = torch.cat([torch.arange(n[i], device=z.device) for i in range(n.shape[0])], dim=0)
        pos = self.pos_gen(k)  # total_nodes x max_n

        keys = self.key_net(pos)

        vals_rep = torch.repeat_interleave(z, n, dim=0)
        zp = vals_rep * keys

        x = self.decoder(zp)
        batch = torch.repeat_interleave(torch.arange(n.shape[0], device=z.device), n, dim=0)

        self.x = x
        self.batch = batch
        return x, batch

    def get_batch_pred(self):
        'Returns: the batch idxs of the outputs x (shape: noutputs)'
        return self.batch

    def get_x_pred(self):
        'Returns: the outputs x (shape: noutputs x d)'
        return self.x

    def get_n_pred_logits(self):
        'Returns: the class logits for each possible n, up to max_n (shape: batch x max_n)'
        return self.n_pred_logits

    def get_n_pred(self):
        'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
        return self.n_pred


if __name__ == '__main__':

    dim = 3
    max_n = 5
    batch_size = 16

    enc = Encoder(dim=dim)
    dec = Decoder(dim=dim)

    data_list = []
    batch_list = []
    for i in range(batch_size):
        n = torch.randint(low=1, high=max_n, size=(1,))
        x = torch.randn(n[0], dim)
        data_list.append(x)
        batch_list.append(torch.ones(n) * i)
    data = torch.cat(data_list, dim=0)
    batch = torch.cat(batch_list, dim=0).int()

    z = enc(data, batch)
    xr, batchr = dec(z)

    print(x.shape, xr.shape)
    print(batch.shape, batchr.shape)
