import torch

from sae.mlp import build_mlp


class MLPAE(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def forward(self, x, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, None


class Encoder(torch.nn.Module):
    def __init__(self, dim, hidden_dim, n_agents, **kwargs):
        super().__init__()

        self.input_dim = dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        self.encoder_layer = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            nlayers=2,
            midmult=1.,
            layernorm=True,
            nonlinearity=torch.nn.Mish
        )

    def forward(self, x, **kwargs):
        x = x.reshape(x.shape[0] // self.n_agents, -1)
        z = self.encoder_layer(x)
        return z


class Decoder(torch.nn.Module):
    def __init__(self, dim, hidden_dim, n_agents, **kwargs):
        super().__init__()

        self.output_dim = dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        self.decoder_layer = build_mlp(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            nlayers=2,
            midmult=1.,
            layernorm=True,
            nonlinearity=torch.nn.Mish
        )

    def forward(self, x, **kwargs):
        z = self.decoder_layer(x)
        z = z.reshape(x.shape[0] * self.n_agents, -1)
        return z
