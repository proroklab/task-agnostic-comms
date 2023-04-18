import argparse
import time

import torch
from vmas import make_env

from config import Config
from scenario_config import SCENARIO_CONFIG


def _generate_random_action(previous_act, n_actions, num_envs, drift=0.8):
    eps = torch.rand((num_envs, 1))
    rand_action = torch.randint(low=0, high=n_actions + 1, size=(num_envs, 1))

    if previous_act is None:
        return rand_action
    else:
        new_act = previous_act.clone()
        modify = eps > drift
        new_act[modify] = rand_action[modify]
        return new_act


def _generate_random_action_cont(previous_act, action_space, num_envs, drift=0.8):
    rand_action = torch.tensor([action_space.sample() for _ in range(num_envs)])
    if previous_act is None:
        return rand_action
    else:
        new_act = previous_act.clone()
        new_act += torch.normal(0.0, 0.1, size=new_act.shape)
        if action_space.contains(new_act) is False:
            return rand_action
        else:
            return new_act


def sample(
        scenario_name,
        random_obs,
        steps,
        num_envs,
        render
):
    init_time = time.time()

    num_agents = SCENARIO_CONFIG[scenario_name]["num_agents"]
    reset_after = SCENARIO_CONFIG[scenario_name]["reset_after"]

    # Construct VMAS environment
    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=Config.device,
        continuous_actions=False,
        n_agents=num_agents,
    )

    obs_size = env.observation_space[0].shape[0]
    num_actions = env.action_space[0].n - 1
    num_envs = 1 if random_obs else num_envs,

    agent_observations = torch.empty((
        steps,
        num_agents,
        num_envs,
        obs_size
    ))

    if random_obs:
        import numpy as np
        for s in range(steps):
            obs = torch.tensor(np.array(env.observation_space.sample())).unsqueeze(1)
            agent_observations[s] = obs
            if s % 100 == 0:
                print(f"{s}/{steps}")
    else:
        prev_act = [None for _ in range(num_agents)]
        for s in range(steps):

            # Generate action
            actions = []
            for i in range(num_agents):
                # act = _generate_random_action_cont(prev_act[i], env.action_space[i], num_envs)
                act = _generate_random_action(prev_act[i], num_actions, num_envs)
                actions.append(act)
                prev_act[i] = act

            obs, _, dones, _ = env.step(actions)

            agent_observations[s] = torch.stack(obs)

            # Reset environments that are done
            if torch.all(dones):
                env.reset()
            else:
                for i, done in enumerate(dones):
                    if done.item() is True:
                        env.reset_at(i)

            # Reset all environments after a while to ensure we don't sample crazily out-of-distribution
            # e.g. if agents travel outside usual bounds
            if reset_after is not None:
                if s % reset_after == 0:
                    env.reset()

            if render:
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )

            if s % 10 == 0:
                print(f"{s}/{steps}")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(agent_observations, f'samples/{scenario_name}_{timestr}.pt')
    print(f"Saved {agent_observations.shape} observations as {scenario_name}_{timestr}.pt")

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {steps} steps of {num_envs} parallel environments on device {Config.device}"
    )


if __name__ == "__main__":
    # Parse sampling arguments
    parser = argparse.ArgumentParser(prog='Sample observations randomly from VMAS scenarios')
    parser.add_argument('-c', '--scenario', default=None, help='VMAS scenario')
    parser.add_argument('-r', '--random', action='store_true', default=False, help='Sample randomly directly from observation space')
    parser.add_argument('--steps', default=200, type=int, help='number of sampling steps')
    parser.add_argument('--num_envs', default=32, type=int, help='vectorized environments to sample from')
    parser.add_argument('--render', action='store_true', default=False, help='render scenario while sampling')
    parser.add_argument('-d', '--device', default='cuda')
    args = parser.parse_args()

    # Set global configuration
    Config.device = args.device

    sample(
        args.scenario,
        args.random,
        args.steps,
        args.num_envs,
        args.render,
    )
