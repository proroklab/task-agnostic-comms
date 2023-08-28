import argparse
import time

import torch
from torch.optim import Adam

from config import Config
from sae.mlpae import MLPAE
from scenario_config import SCENARIO_CONFIG
from sae.model import AutoEncoder as SAE


def _load_data(data_file, scenario_name, time_str, use_proj, no_stand):
    print(f"Loading {data_file}...")

    # Load file containing all the observations to encode. We expect that
    # the file contains a tensor of shape (steps, agents, envs, obs shape)
    data = torch.load(data_file).to(Config.device).permute(0, 2, 1, 3)

    # We want: (samples, agents, obs shape) so that we can learn to encode
    # all four observations from the agents in one time step.
    data = torch.flatten(data, start_dim=0, end_dim=1)

    # Shuffle the data (but only in the first dimension)
    data = data[torch.randperm(data.size()[0])]

    # Standardise data
    data /= 5.0

    print("Loaded data with shape", data.shape)

    return data


def _train_test_split(data, train_proportion, test_lim):
    n_train_samples = int(len(data) * train_proportion)
    n_test_samples = min(len(data) - n_train_samples, test_lim)
    n_train_samples = len(data) - n_test_samples

    return data[:n_train_samples], data[n_train_samples: n_train_samples + n_test_samples]


def train(
        scenario_name,
        data_file,
        model_type,
        use_proj,
        no_stand,
        latent_dim,
        batches_per_epoch=256,
        test_lim=1024
):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    set_size = SCENARIO_CONFIG[scenario_name]["num_agents"]

    # Load and process data
    data = _load_data(data_file, scenario_name, time_str, use_proj, no_stand)
    train_data, test_data = _train_test_split(data, train_proportion=0.8, test_lim=test_lim)

    # Flatten first two dimensions to put samples and agents together to get [samples, obs_dim]
    # as agents will be accounted for by the batch index
    train_data = torch.flatten(train_data, start_dim=0, end_dim=1).to(Config.device)
    test_data = torch.flatten(test_data, start_dim=0, end_dim=1).to(Config.device)

    batch_train = torch.arange(batches_per_epoch // set_size, device=Config.device).repeat_interleave(set_size)
    batch_test = torch.arange(test_data.shape[0] // set_size, device=Config.device).repeat_interleave(set_size)

    # Construct the autoencoder
    model_dim = data.shape[-1]
    if model_type == "sae":
        autoencoder = SAE(dim=model_dim, hidden_dim=latent_dim).to(Config.device)
    else:
        autoencoder = MLPAE(dim=model_dim, hidden_dim=latent_dim, n_agents=set_size).to(Config.device)
    optimizer = Adam(autoencoder.parameters())

    epochs = len(train_data) // batches_per_epoch

    print(f"Training model using device {Config.device} for {epochs} epochs")

    import wandb
    run = wandb.init(
        project="acs-project",
        entity="dhjayalath",
        name="train_sae",
        sync_tensorboard=True,
        config={
            "epochs": epochs,
            "train_size": len(train_data),
            "test_size": len(test_data),
        }
    )

    for epoch in range(epochs):

        optimizer.zero_grad()

        x = train_data[epoch * batches_per_epoch: (epoch + 1) * batches_per_epoch]
        xr, _ = autoencoder(x, batch=batch_train)

        if model_type == "sae":
            train_loss_vars = autoencoder.loss()
            sae_loss = train_loss_vars["loss"]
            sae_loss.backward()
        else:
            mse_loss = torch.nn.functional.mse_loss(x, xr)

        optimizer.step()

        with torch.no_grad():

            xr, _ = autoencoder(test_data, batch=batch_test)

            if model_type == "sae":
                test_loss_vars = autoencoder.loss()
            else:
                test_mse_loss = torch.nn.functional.mse_loss(test_data, xr)

            if epoch % 1000 == 0:

                print("\t Epoch", epoch)
                if model_type == "sae":
                    print("----- TRAIN -----")
                    print(train_loss_vars)
                    print("----- TEST -----")
                    print(test_loss_vars)
                else:
                    print("----- TRAIN -----")
                    print(mse_loss)
                    print("----- TEST -----")
                    print(test_mse_loss)

                if model_type == "sae":
                    if len(xr) >= set_size:
                        inputs_sorted = autoencoder.encoder.get_x()
                        print("Length (source, recon)", inputs_sorted[0:set_size].shape, xr[0:set_size].shape)
                        print("Source", inputs_sorted[0:set_size])
                        print("Recon", xr[0:set_size])
                else:
                    print("Source", test_data[0:set_size])
                    print("Recon", xr[0:set_size])

                if epoch % 2000 == 0 and epoch != 0:
                    time_str = time.strftime("%Y%m%d-%H%M%S")
                    file_str = f"weights/{model_type}_{scenario_name}_{epoch}_{time_str}.pt"
                    torch.save(autoencoder.state_dict(), file_str)

            wandb.log({
                          "train_loss": train_loss_vars["loss"],
                          "mse_loss": train_loss_vars["mse_loss"],
                          "size_loss": train_loss_vars["size_loss"],
                          "corr": train_loss_vars["corr"],
                          "test_loss": test_loss_vars["loss"],
                          "test_mse_loss": test_loss_vars["mse_loss"],
                          "test_size_loss": test_loss_vars["size_loss"],
                          "test_corr": test_loss_vars["corr"],
                      } if model_type == "sae" else {
                "train_loss": mse_loss,
                "test_loss": test_mse_loss,
            })

    run.finish()

    # Show an example reconstruction
    print("Showing reconstruction on random sample...")
    with torch.no_grad():

        xr, _ = autoencoder(test_data, batch=batch_test)
        inputs_sorted = autoencoder.encoder.get_x()
        print("Length (source, recon)", inputs_sorted[0].shape, xr[0].shape)
        print("Source", inputs_sorted[0:set_size])
        print("Recon", xr[0:set_size])

    # Save model
    file_str = f"weights/{model_type}_{scenario_name}_{time_str}.pt"
    torch.save(autoencoder.state_dict(), file_str)
    print(f"Saved model to {file_str}")


if __name__ == "__main__":
    # Parse autoencoder training arguments
    parser = argparse.ArgumentParser(prog='Train SAE on sampled data')
    parser.add_argument('--latent', default=16, type=int, help='latent dimension of set autoencoder to use')
    parser.add_argument('--data', help='file to load for training data (sampled observations)')
    parser.add_argument('--ae_type', default='sae', help='select autoencoder type: sae/mlp')
    parser.add_argument('--use_proj', action='store_true', default=False, help='project observations into high-dimensional space')
    parser.add_argument('--no_stand', action='store_true', default=False, help='do not standardise inputs')

    parser.add_argument('-c', '--scenario', default=None, help='VMAS scenario')
    parser.add_argument('-d', '--device', default='cuda')
    args = parser.parse_args()

    # Set global configuration
    Config.device = args.device

    train(
        args.scenario,
        args.data,
        args.ae_type,
        args.use_proj,
        args.no_stand,
        args.latent,
    )
