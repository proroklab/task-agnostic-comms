# Generalising Multi-Agent Cooperation through Task-Agnostic Communication

## Setup

Please make sure you have:
- Python 3.10+
- CUDA GPU(s)

Start by cloning the repository and submodules, then enter the top-level directory:

`git clone --recurse-submodules <>.git && cd marl-comms-optimisation`

Set `WANDB_ENTITY` and `WANDB_PROJECT` for `wandb` logging in `config.py`

Each MARL suite requires its own setup. Use the relevant virtual environment for running the associated experiments.

### Melting Pot Setup
1. Change directory to the Melting Pot folder
    
    `cd meltingpot-marlcomms`
2. Create a new virtual environment
    
    `python -m venv mp_env && source mp_venv/bin/activate`
3. Install dependencies
    
    `pip install -r requirements_mp.txt`
4. Create directories to store weights and samples
    
    `mkdir weights && mkdir samples`

## Running VMAS Experiments

### VMAS Setup
1. Create a new virtual environment
    
    `python -m venv vmas_env && source vmas_env/bin/activate`
2. Install dependencies
    
    `pip install -r requirements_vmas.txt`
3. Create directories to store weights and samples
    
    `mkdir weights && mkdir samples`

In the sections that follow, substitute `<task>` with one of `norm_discovery, norm_swarm, norm_flocking`.

### Collecting pre-training data

Uniformly randomly sample 1M observations from the observation space of Discovery or Swarm.

`python sample_vmas.py --scenario <task> --steps 1000000 --random --device cpu`

The results will be saved to `samples/<task>_<time>.pt`

Note: You can modify the number of agents you collect results for using the relevant task entry in `scenario_config.py`.

### Training a task-agnostic comms strategy offline
Note: `--latent` is the expected latent dimension of the set autoencoder. To reproduce our work, this should be equal `agent observation dim * no. of agents`.

`python train_sae.py --scenario <task> --latent <> --data samples/<task>_<time>.pt`

A state dict is saved every 2000 steps to `weights/sae_<task>_<epoch>_<time>.pt` and `weights/sae_<task>_latest.pt`.

Note: if you have multiple samples (e.g. if you want to train a strategy for 1, 2, and 3 agents), then use `train_sae_scaling.py` instead, specifying a list like `--data sample1.pt sample2.pt sample3.pt`.

### Training a task-specific comms strategy on-policy
Note: `--pisa_dim <>` is the dimension of each individual agent observation expected by the autoencoder.

`python policy.py --train_specific --scenario <task> --pisa_dim <> --seed <> --home`

As this policy trains, the learned task-specific communication strategy will be saved to `weights/sae_policy_wk<worker_index>_<time>.pt` every `--eval_interval` steps.

### Testing comms strategies on-policy

Training stats should be saved to `~/ray_results` where they can be viewed with `tensorboard --logdir ~/ray_results`

No-comms:

`python policy.py --no_comms --scenario <task> --pisa_dim <> --seed <> --home`

Task-agnostic:

`python policy.py --task_agnostic --scenario <task> --pisa_dim <> --pisa_path weights/sae_<task>_latest.pt --seed <> --home`

Task-specific:

`python policy.py --task_specific --scenario <task> --pisa_dim <> --pisa_path weights/sae_policy_wk<worker_index>_<time>.pt --seed <> --home`

Note: If you wish to train a policy with a different number of agents than the default (e.g. to test out-of-distribution performance of comms strategies), then specify `--scaling_agents <>` with the number of agents you wish to use.

## Running Melting Pot Experiments

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