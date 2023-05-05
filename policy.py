import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID, AgentID

from model_ippo import PolicyIPPO
from model_cppo import PolicyCPPO
from model_hetippo import PolicyHetIPPO
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
            sae = policies["default_policy"].model.autoencoder
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
        n_batches = obs.shape[0]

        obs = obs.reshape(n_batches, pi.model.n_agents, -1)

        if pi.model.use_proj is True:
            obs = obs @ pi.model.proj

        if pi.model.no_stand is False:
            obs = (obs - pi.model.data_mean) / pi.model.data_std
            obs = torch.nan_to_num(
                obs, nan=0.0, posinf=0.0, neginf=0.0
            )  # Replace NaNs introduced by zero-division with zero

        obs = torch.flatten(obs, start_dim=0, end_dim=1)  # [batches * agents, obs_size]
        batch = torch.arange(n_batches, device=obs.device).repeat_interleave(pi.model.n_agents)

        sae = pi.model.autoencoder
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


def policy(
        scenario_name,
        model,
        encoder,
        encoding_dim,
        encoder_file,
        encoder_loss,
        use_proj,
        no_stand,
        train_batch_size,
        sgd_minibatch_size,
        max_steps,
        training_iterations,
        num_workers,
        num_envs,
        num_cpus_per_worker,
        seed,
        render_env,
        wandb_name,
        no_render_eval_callbacks,
        vmas_device="cpu",
):
    num_envs_per_worker = num_envs
    rollout_fragment_length = 125

    if model == 'ippo':
        ModelCatalog.register_custom_model("policy_net", PolicyIPPO)
    elif model == 'cppo':
        ModelCatalog.register_custom_model("policy_net", PolicyCPPO)
    elif model == 'hetippo':
        ModelCatalog.register_custom_model("policy_net", PolicyHetIPPO)
    elif model == 'joippo':
        ModelCatalog.register_custom_model("policy_net", PolicyJOIPPO)
    else:
        raise AssertionError

    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )

    if no_render_eval_callbacks is True:
        callbacks = []
    else:
        callbacks = [RenderingCallbacks, EvaluationCallbacks]

    if model == "joippo" and encoder is not None:
        callbacks.insert(0, ReconstructionLossCallbacks)
    if encoder_loss == 'policy':
        callbacks.insert(0, SAECheckpointCallbacks)

    if not ray.is_initialized():
        ray.init()
        print("Ray init!")

    if "pursuit" in scenario_name:
        from pettingzoo.sisl import pursuit_v4
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        pz_env_creator = lambda config: pursuit_v4.env(
            max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=30,
            n_pursuers=8, obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
            catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
        register_env(scenario_name, lambda config: ParallelPettingZooEnv(pz_env_creator(config)))
    else:
        register_env(scenario_name, lambda config: env_creator(config))

    if Config.device == 'cuda':
        num_gpus = 1  # Driver GPU
        num_gpus_per_worker = 0  # VMAS will be on CPU
    else:
        num_gpus = 0
        num_gpus_per_worker = 0

    print("VMAS Device", vmas_device, "rllib GPUs", num_gpus, "rllib GPUs/worker", num_gpus_per_worker)

    # Train policy!
    ray.tune.run(
        MultiPPOTrainer,
        stop={"training_iteration": training_iterations},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        callbacks=[
            WandbLoggerCallback(
                project=f"acs_project",
                name=f"{wandb_name}-seed-{seed}",
                entity="dhjayalath",
                api_key="",
            )
        ],
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "render_env": render_env,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0.01,
            "train_batch_size": train_batch_size,
            # Should remain close to max steps to avoid bias
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": 45,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            # "num_gpus_per_worker": num_gpus_per_worker,
            # "num_cpus_per_worker": num_cpus_per_worker,
            "num_envs_per_worker": num_envs_per_worker,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "complete_episodes",
            "model": {
                "custom_model": "policy_net",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "scenario_name": scenario_name,
                    "encoder": encoder,
                    "encoding_dim": encoding_dim,
                    "encoder_file": os.path.abspath(encoder_file) if encoder_file is not None else encoder_file,
                    "encoder_loss": encoder_loss,
                    "use_proj": use_proj,
                    "no_stand": no_stand,
                    "cwd": os.getcwd(),
                    "core_hidden_dim": 256,
                    "head_hidden_dim": 32,
                    "wandb_grouping": wandb_name,
                },
            },
            "env_config": {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": True,
                "max_steps": max_steps,
                "share_reward": True,
                # Scenario specific variables
                "scenario_config": {
                    "n_agents": SCENARIO_CONFIG[scenario_name]["num_agents"],
                },
            },
            "evaluation_interval": 10,  # TODO: Change to 10
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,  # Will this do the trick?
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks(callbacks),  # Removed RenderingCallbacks
            },
            "callbacks": EvaluationCallbacks,
        },
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train policy with SAE')

    # Required
    parser.add_argument('-c', '--scenario', default=None, help='VMAS scenario')
    parser.add_argument('--model', default='ippo', help='Model: ippo/cppo/hetippo/joippo')

    # Joint observations with encoder
    parser.add_argument('--encoder', default=None, help='Encoder type: mlp/sae. Do not use this option for None')
    parser.add_argument('--encoding_dim', default=None, type=int, help='Encoding dimension')
    parser.add_argument('--encoder_loss', default=None, help='Train encoder loss: policy/recon leave None for frozen')
    parser.add_argument('--encoder_file', default=None, help='File with encoder weights')

    # Misc.
    parser.add_argument('--use_proj', action="store_true", default=False, help='project observations into higher space')
    parser.add_argument('--no_stand', action="store_true", default=False, help='do not standardise observations')
    parser.add_argument('--no_render_eval_callbacks', action="store_true", default=False, help='disable render and eval callbacks for HPC')

    # Optional
    parser.add_argument('--render', action="store_true", default=False, help='Render environment')
    parser.add_argument('--train_batch_size', default=60000, type=int, help='train batch size')
    parser.add_argument('--sgd_minibatch_size', default=4096, type=int, help='sgd minibatch size')
    parser.add_argument('--training_iterations', default=5000, type=int, help='number of training iterations')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb_name', default="rllib_training", help='wandb run name')


    parser.add_argument('--num_envs', default=32, type=int)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--num_cpus_per_worker', default=1, type=int)
    parser.add_argument('-d', '--device', default='cuda')
    args = parser.parse_args()

    # Set global configuration
    Config.device = args.device

    policy(
        scenario_name=args.scenario,
        model=args.model,
        encoder=args.encoder,
        encoding_dim=args.encoding_dim,
        encoder_file=args.encoder_file,
        encoder_loss=args.encoder_loss,
        use_proj=args.use_proj,
        no_stand=args.no_stand,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
        max_steps=SCENARIO_CONFIG[args.scenario]["max_steps"],
        training_iterations=args.training_iterations,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        render_env=args.render,
        wandb_name=args.wandb_name,
        no_render_eval_callbacks=args.no_render_eval_callbacks,
        seed=args.seed,
    )
