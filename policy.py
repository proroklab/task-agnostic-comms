import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID, AgentID

from model_joippo import PolicyJOIPPO

from multi_action_dist import TorchHomogeneousMultiActionDistribution
from multi_trainer import MultiPPOTrainer

from scenario_config import SCENARIO_CONFIG
import wandb
from ray.rllib import BaseEnv, RolloutWorker, Policy, SampleBatch
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from vmas import make_env, Wrapper

from config import Config

import time
import torch


class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            episode: MultiAgentEpisode,
            **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                try:
                    episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                except KeyError:
                    episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: MultiAgentEpisode,
            **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()


class RenderingCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Episode,
            **kwargs,
    ) -> None:
        self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
    ) -> None:
        vid = np.transpose(self.frames, (0, 3, 1, 2))
        episode.media["rendering"] = wandb.Video(
            vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
        )
        self.frames = []


class SAECheckpointCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
    ) -> None:
        if worker.worker_index == 1:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            sae = policies["default_policy"].model.pisa
            file_str = f"weights/sae_policy_wk{worker.worker_index}_{time_str}.pt"
            torch.save(sae, file_str)
            print(f"Saved SAE trained with policy losses to {file_str}")


class ReconstructionLossCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        pi = policies["default_policy"]
        obs = torch.tensor(postprocessed_batch["obs"]).clone()

        # Standardise
        obs /= 5.0

        n_batches = obs.shape[0]

        obs = obs.reshape(n_batches, pi.model.n_agents, -1)

        obs = torch.flatten(obs, start_dim=0, end_dim=1)  # [batches * agents, obs_size]
        batch = torch.arange(n_batches, device=obs.device).repeat_interleave(pi.model.n_agents)

        sae = pi.model.pisa
        sae(obs, batch=batch)
        losses = sae.loss()
        sae_loss = losses["loss"]
        recon_loss = losses["mse_loss"]

        if torch.is_tensor(sae_loss):
            sae_loss = sae_loss.item()
        if torch.is_tensor(recon_loss):
            recon_loss = recon_loss.item()

        # Add if not worker or create if it is. Tracks running means.
        episode.custom_metrics[f"sae_loss"] = sae_loss
        episode.custom_metrics[f"recon_loss"] = recon_loss


# VMAS environment creator
def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env

def setup_callbacks(**kwargs):
    if kwargs["excalibur"] or kwargs["merlin"] or kwargs["safe"]:
        callbacks = []
        if not kwargs["no_comms"]:
            # Log AE / PISA loss when they are being used
            callbacks.insert(0, ReconstructionLossCallbacks)
        if kwargs["train_specific"]:
            # Checkpoint PISA when trained with policy loss
            callbacks.insert(0, SAECheckpointCallbacks)
        return callbacks
    else:
        callbacks = [RenderingCallbacks]
        if not kwargs["no_comms"]:
            # Log AE / PISA loss when they are being used
            callbacks.insert(0, ReconstructionLossCallbacks)
        if kwargs["train_specific"]:
            # Checkpoint PISA when trained with policy loss
            callbacks.insert(0, SAECheckpointCallbacks)
        return callbacks

def policy(**kwargs):

    ModelCatalog.register_custom_model("policy_net", PolicyJOIPPO)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )
    callbacks = setup_callbacks(**kwargs)

    if not ray.is_initialized():
        if kwargs["excalibur"] or kwargs["merlin"]:
            ray.init(address="auto")
        else:
            ray.init()
        print("ray intialised.")

    register_env(kwargs["scenario"], lambda config: env_creator(config))

    if Config.device == 'cuda':
        num_gpus = 1  # Driver GPU
        num_gpus_per_worker = 0  # VMAS will be on CPU
    else:
        num_gpus = 0
        num_gpus_per_worker = 0

    # Determine mode
    if kwargs["task_agnostic"]:
        mode = "task_agnostic"
    elif kwargs["task_specific"]:
        mode = "task_specific"
    elif kwargs["train_specific"]:
        mode = "train_specific"
    else:
        mode = "no_comms"

    print("\n\n-----------------------------------------------------------\n\n")
    print(f"experiment type = {mode}")
    print(f"device = {Config.device}")
    print(f"scenario = {kwargs['scenario']}")
    print(f"seed = {kwargs['seed']}")
    print(f"pisa path = {kwargs['pisa_path']}")
    print(f"pisa latent dim = {kwargs['pisa_dim']}")
    print(f"excalibur = {kwargs['excalibur']}")
    print(f"merlin = {kwargs['merlin']}")
    print("\n\n-----------------------------------------------------------\n\n")

    if kwargs["excalibur"]:
        local_dir = "/local/scratch-2/dhj26/ray_results"
    elif kwargs["merlin"]:
        local_dir = "/local/scratch/dhj26/ray_results"
    elif kwargs["home"]:
        local_dir = "~/ray_results"
    else:
        local_dir = "/rds/user/dhj26/hpc-work/ray_results"

    # Train policy!
    ray.tune.run(
        MultiPPOTrainer,
        local_dir=local_dir,
        name=f"PPO_{time.strftime('%Y%m%d-%H%M%S')}",
        stop={"training_iteration": kwargs["training_iterations"]},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        callbacks=[
            WandbLoggerCallback(
                project=f"task-agnostic-comms",
                name=f"{kwargs['scenario']}+{mode}+{kwargs['seed']}",
                entity="dhjayalath",
                api_key="",
            )
        ],
        config={
            "seed": kwargs["seed"],
            "framework": "torch",
            "env": kwargs["scenario"],
            "render_env": False,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": kwargs["train_batch_size"],
            # Should remain close to max steps to avoid bias
            "rollout_fragment_length": kwargs["rollout_fragment_length"],
            "sgd_minibatch_size": kwargs["sgd_minibatch_size"],
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": kwargs["num_workers"],
            "num_envs_per_worker": kwargs["num_envs"],
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "model": {
                "custom_model": "policy_net",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    **kwargs,
                    "pisa_path": os.path.abspath(kwargs["pisa_path"]) if kwargs["pisa_path"] is not None else kwargs["pisa_path"],
                    "wandb_grouping": f"{kwargs['scenario']}+{mode}",
                },
            },
            "env_config": {
                "device": "cpu",
                "num_envs": kwargs["num_envs"],
                "scenario_name": kwargs["scenario"],
                "continuous_actions": False,
                "max_steps": kwargs["max_steps"],
                "share_reward": True,
                # Scenario specific variables
                "scenario_config": {
                    "n_agents": SCENARIO_CONFIG[kwargs["scenario"]]["num_agents"] if kwargs["scaling_agents"] is None else kwargs["scaling_agents"],
                },
            },
            "evaluation_interval": kwargs["eval_interval"],
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks(callbacks),
            },
        },
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train policy with SAE')

    # Modes
    parser.add_argument('--task_agnostic', action='store_true', default=False, help='Task-agnostic pre-trained PISA experiment')
    parser.add_argument('--task_specific', action='store_true', default=False, help='Reused pre-trained PISA experiment')
    parser.add_argument('--train_specific', action='store_true', default=False, help='Train PISA with policy losses experiment')
    parser.add_argument('--no_comms', action='store_true', default=False, help='No communications experiment')

    # Required
    parser.add_argument('--scenario', type=str, default=None, help='MeltingPot scenario')
    parser.add_argument('--pisa_dim', type=int, default=None, help='PISA latent state dimensionality') # FIXME: Is this required? Can't we infer it?
    parser.add_argument('--pisa_path', type=str, default=None, help='Path to PISA autoencoder state dict')
    parser.add_argument('--seed', type=int, default=None)

    # Optional
    parser.add_argument('--scaling_agents', default=None, type=int, help='Use a different number of agents to the default for scaling')
    parser.add_argument('--policy_width', default=256, type=int, help='Policy network width')
    parser.add_argument('--excalibur', action='store_true', default=False, help='Disable callbacks for compatibility on excalibur/HPC')
    parser.add_argument('--merlin', action='store_true', default=False)
    parser.add_argument('--home', action='store_true', default=False)
    parser.add_argument('--safe', action='store_true', default=False)
    parser.add_argument('--train_batch_size', default=60000, type=int, help='train batch size')
    parser.add_argument('--sgd_minibatch_size', default=4096, type=int, help='sgd minibatch size')
    parser.add_argument('--training_iterations', default=500, type=int, help='number of training iterations')
    parser.add_argument('--rollout_fragment_length', default=125, type=int, help='Rollout fragment length')
    parser.add_argument('--eval_interval', default=10, type=int, help='Evaluation interval')
    parser.add_argument('--num_envs', default=32, type=int)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--num_cpus_per_worker', default=1, type=int)
    parser.add_argument('-d', '--device', default='cuda')

    args = parser.parse_args()

    # Check valid argument configuration
    assert args.task_agnostic or args.task_specific or args.train_specific or args.no_comms, "No experiment mode specified"
    assert args.scenario is not None, "--scenario not specified"
    assert args.pisa_dim is not None, "--pisa_dim not specified"
    assert args.seed is not None, "--seed not specified"
    if args.task_agnostic or args.task_specific:
        assert args.pisa_path, "--pisa_path not specified"

    # Set global configuration
    Config.device = args.device

    policy(max_steps=SCENARIO_CONFIG[args.scenario]["max_steps"], **vars(args))
