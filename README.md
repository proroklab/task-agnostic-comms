# Generalising Multi-Agent Cooperation through Task-Agnostic Communication

## Setup

Please make sure you have:
- Python 3.10+
- CUDA GPU(s)

Start by cloning the repository and submodules, then enter the top-level directory:

`git clone --recurse-submodules <>.git && cd marl-comms-optimisation`

Set `WANDB_ENTITY` and `WANDB_PROJECT` for `wandb` logging in `config.py`

Each MARL suite requires its own setup. Use the relevant virtual environment for running the associated experiments.

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
The instruction for this section are very similar to running the VMAS experiments above.

### Melting Pot Setup
1. Change directory to the Melting Pot folder
    
    `cd meltingpot-marlcomms`
2. Create a new virtual environment
    
    `python -m venv mp_env && source mp_venv/bin/activate`
3. Install dependencies
    
    `pip install -r requirements_mp.txt && pip install -e .`
4. Create directories to store weights and samples
    
    `mkdir weights && mkdir samples`

### Collecting pre-training data

Uniformly randomly sample 1M observations from the observation space of Discovery or Swarm.

`python sample_meltingpot.py --scenario <task> --steps 1000000`

The results will be saved to `samples/<task>_<time>.pt`

### Training the image encoder

`python train_cnn.py --scenario <task> --data_path samples/<task>_<time>.pt --image_width <>`

Weights will be saved periodically to `weights/cnn_<task>_best.pt`.

### Training a task-agnostic comms strategy offline

Note: `--latent` is the expected latent dimension of the set autoencoder. To reproduce our work, this should be equal `agent observation dim * no. of agents`.

`python train_pisa.py --scenario <task> --data_path samples/<task>_<time>.pt --cnn_path weights/cnn_<task>_best.pt`

Weights will be saved periodically to `weights/pisa_<task>_best.pt`

### Training a task-specific comms strategy on-policy
Note: `--pisa_dim <>` is the dimension of each individual agent observation expected by the set autoencoder.

`python policy_mp.py --train_specific --scenario <task> --cnn_path weights/cnn_<task>_best.pt --pisa_dim <> --seed <>`

As this policy trains, the learned task-specific communication strategy will be saved to `weights/pisa_policy_wk<worker_index>_<time>.pt` every `eval_interval` steps.

### Testing comms strategies on-policy

Training stats should be saved to `~/ray_results` where they can be viewed with `tensorboard --logdir ~/ray_results`

No-comms:

`python policy_mp.py --no_comms --scenario <task> --cnn_path weights/cnn_<task>_best.pt --pisa_dim <> --seed <>`

Task-agnostic:

`python policy_mp.py --task_agnostic --scenario <task> --cnn_path weights/cnn_<task>_best.pt --pisa_dim <> --pisa_path weights/pisa_<task>_best.pt --seed <>`

Task-specific:

`python policy_mp.py --task_specific --scenario <task> --cnn_path weights/cnn_<task>_best.pt --pisa_dim <> --pisa_path weights/pisa_policy_wk<worker_index>_<time>.pt --seed <>`

## Troubleshooting
If you have problems installing `dmlab2d`, please see instructions at https://github.com/google-deepmind/lab2d to build it from source.