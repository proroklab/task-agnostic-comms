# Reconstructing Markov States in Decentralised MARL (Base repository)

## VMAS Instructions
### Setup
Clone this repository and follow the steps below.
```bash
# 1. Create a Python 3.9+ virtual environment
virtualenv env && source env/bin/activate && pip install --upgrade pip

# 2. Install required packages
pip install ray[rllib]==2.1.0 torch torchvision numpy==1.23.5 moviepy imageio wandb git+https://github.com/proroklab/VectorizedMultiAgentSimulator.git && pip uninstall grpcio && pip install grpcio==1.32.0

# 3. Create directory to store weights and samples and log in to wandb 
mkdir weights && mkdir samples && wandb login
```

### Collecting samples
The following command will sample observations randomly from the scenario and save them to a file under `samples/*.pt`.
```bash
python sample_vmas.py -c <scenario name> --steps 100000 -d cpu --continuous
```

### Pre-training the set autoencoder
This command will load the sampled observations from disk and pre-train a set autoencoder using them. The weights for
the pre-trained autoencoder will be saved to `weights/*.pt`
```bash
python train_sae.py -c <scenario name> --data <path to samples> --latent <latent dimension> -d <device>
```

### Training the policy
Edit and execute either `train_flocking.sh` or `train_discovery.sh`.

## Melting Pot Instructions

### Setup
```bash
# 1. Create a Python 3.9+ virtual environment
cd meltingpot-marlcomms && virtualenv -p /usr/bin/python3.9 mp_venv && source mp_venv/bin/activate && pip install --upgrade pip

# 2. Install required packages (if there are issues installing Melting Pot, do so manually following their instructions)
pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl && pip install -e . && pip install "ray[rllib]"==2.3.1 wandb imageio moviepy torch torchvision gym

# 3. Create required directories and log in to wandb
mkdir weights && mkdir samples && wandb login
```

### Collecting samples
Sampled pixel data will be saved into `samples/<subsrate>/<set>/*.png`
```bash
python sample_meltingpot.py -c <substrate name> --steps 100000 -d <device>
```

### Pre-training the pixel set autoencoder
```bash
# 1. Train the convolutional autoencoder (saves state dict. to weights/*.pt)
python train_cnn.py --scenario <substrate name> --data_path <path to samples> --image_width <88 or 40> --latent_dim <latent dim> --device <device>

# 2. Train the set autoencoder (saves state dict. to weights/*.pt)
python train_pisa.py --scenario <substrate name> --data_path <path to samples> --cnn_path <path to Conv. AE state dict.> --image_width <88 or 40> --data_dim <latent dim of Conv. AE> --device <device>
```

### Training the policy
Edit and execute any of the `train_<substrate>.sh` scripts.