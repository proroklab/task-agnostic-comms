from torch import nn
from sae.model import AutoEncoder as SAE

class ConvCAESAE(nn.Module):
    def __init__(self, dim=768, hidden_dim=100):
        super().__init__()
        self.cnn = CNNAE(latent_dim=hidden_dim)
        self.sae = SAE(dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x, batch):
        # x has shape (steps * envs * agents, ...)

        # Encoder with ConvVAE
        mu, logvar = self.cnn.encode(x)  # (steps * agents, 11, 11, 3)
        z = self.cnn.reparameterize(mu, logvar)

        # Encode + Decode with SAE
        zr, _ = self.sae(z, batch=batch)  # (steps * agents, dim) ??

        # Decode SAE reconstruction with ConvVAE
        x_recon = self.cnn.decode(zr)  # (steps * agents, 88, 88, 3)

        # Decode original latent with ConvVAE
        x_cnn_recon = self.cnn.decode(z)

        return x_recon, x_cnn_recon, mu, logvar

    def encode(self, x, batch):
        # Encode with Conv VAE
        mu, logvar = self.cnn.encode(x)  # (steps * agents, 11, 11, 3)
        z = self.cnn.reparameterize(mu, logvar)

        # Encode with SAE
        z_sae = self.sae.encoder(z, batch=batch)

        return z_sae

    def decode(self, z_sae):
        z = self.sae.decoder(z_sae)
        x = self.cnn.decode(z)
        return x

class ConvVAESAE(nn.Module):
    def __init__(self, dim=768, hidden_dim=100, n_agents=2, obs_w=88):
        super().__init__()
        self.cnn = VAE(latent_dim=dim, mu_dim=hidden_dim, obs_w=obs_w)
        self.sae = SAE(dim=hidden_dim, hidden_dim=hidden_dim * n_agents)

    def forward(self, x, batch):
        # x has shape (steps * envs * agents, ...)

        # Encoder with ConvVAE
        mu, logvar = self.cnn.encode(x)  # (steps * agents, 11, 11, 3)
        z = self.cnn.reparameterize(mu, logvar)

        # Encode + Decode with SAE
        zr, _ = self.sae(z, batch=batch)  # (steps * agents, dim) ??

        # Decode SAE reconstruction with ConvVAE
        x_recon = self.cnn.decode(zr)  # (steps * agents, 88, 88, 3)

        # Decode original latent with ConvVAE
        x_cnn_recon = self.cnn.decode(z)

        return x_recon, x_cnn_recon, mu, logvar

    def encode(self, x, batch):
        # Encode with Conv VAE
        mu, logvar = self.cnn.encode(x)  # (steps * agents, 11, 11, 3)
        z = self.cnn.reparameterize(mu, logvar)

        # Encode with SAE
        z_sae = self.sae.encoder(z, batch=batch)

        return z_sae

    def decode(self, z_sae):
        z = self.sae.decoder(z_sae)
        x = self.cnn.decode(z)
        return x

class CNNAE(nn.Module):
    def __init__(self,
                 hidden_channels: int = 32,
                 latent_dim: int = 20,
                 act_fn: object = nn.LeakyReLU):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16, 88x88 => 44x44
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, stride=2),
            # 16x16 => 8x8, 44x44 => 22x22
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, stride=2),
            # 8x8 => 4x4, 22x22 => 11x11
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            # nn.Linear(hidden_channels * 2 * 4 * 4, latent_dim)
            nn.Linear(hidden_channels * 2 * 11 * 11, latent_dim)
        )

        self.decoder = nn.Sequential(
            # Linear
            # nn.Linear(latent_dim, hidden_channels * 2 * 4 * 4),
            nn.Linear(latent_dim, hidden_channels * 2 * 11 * 11),
            act_fn(),
            # Shape
            # nn.Unflatten(1, (2 * hidden_channels, 4, 4)),
            nn.Unflatten(1, (2 * hidden_channels, 11, 11)),
            # CNN
            nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(hidden_channels, 3, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


import torch
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self,
                 hidden_channels: int = 32,
                 latent_dim: int = 384,
                 mu_dim: int = 100,
                 obs_w: int = 88,
                 act_fn: object = nn.LeakyReLU
                 ):
        super(VAE, self).__init__()
        obs_lat = round(obs_w / 8)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16, 88x88 => 44x44
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, stride=2),
            # 16x16 => 8x8, 44x44 => 22x22
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, stride=2),
            # 8x8 => 4x4, 22x22 => 11x11
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            # nn.Linear(hidden_channels * 2 * 4 * 4, latent_dim)
            nn.Linear(hidden_channels * 2 * obs_lat * obs_lat, latent_dim)
        )

        self.decoder = nn.Sequential(
            # Linear
            # nn.Linear(latent_dim, hidden_channels * 2 * 4 * 4),
            nn.Linear(latent_dim, hidden_channels * 2 * obs_lat * obs_lat),
            act_fn(),
            # Shape
            # nn.Unflatten(1, (2 * hidden_channels, 4, 4)),
            nn.Unflatten(1, (2 * hidden_channels, obs_lat, obs_lat)),
            # CNN
            nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(hidden_channels, 3, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            # nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            nn.Sigmoid()
        )

        # self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(latent_dim, mu_dim)
        self.fc22 = nn.Linear(latent_dim, mu_dim)
        self.fc3 = nn.Linear(mu_dim, latent_dim)
        # self.fc4 = nn.Linear(latent_dim, 784)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        return self.decoder(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0 * KLD  # TODO: Change back to 0

    # return BCE + 0.01 * KLD
