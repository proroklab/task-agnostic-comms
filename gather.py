#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Callable, Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        self.collision_reward = kwargs.get("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 1)

        self.plot_grid = True
        self.desired_distance = 0.1
        self.min_collision_distance = 0.005
        self.x_dim = 1
        self.y_dim = 1

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5)

        self._target = Landmark(
            name=f"target",
            collide=False,
            movable=False,
            shape=Sphere(radius=0.05),
            color=Color.GREEN,
        )
        world.add_landmark(self._target)

        goal_entity_filter: Callable[[Entity], bool] = lambda e: not isinstance(
            e, Agent
        ) and not e.name == "target"

        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.2,
                        entity_filter=goal_entity_filter,
                    )
                ],
                render_action=True,
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.dist_rew = agent.collision_rew.clone()

            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

        return world

    def reset_world_at(self, env_index: int = None):
        target_pos = torch.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        )
        target_pos[:, Y] = -self.y_dim
        self._target.set_pos(target_pos, batch_index=env_index)

        ScenarioUtils.spawn_entities_randomly(
            self.obstacles + self.world.policy_agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-self.x_dim, self.x_dim),
            y_bounds=(-self.y_dim, self.y_dim),
        )

        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
            self._done[:] = False
        else:
            self.t[env_index] = 0
            self._done[env_index] = False

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0

        if is_first:
            self.t += 1

            # Check if eaten
            self._done = torch.any(
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - self._target.state.pos, dim=1
                        )
                        < a.shape.radius + self._target.shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )

            # Avoid collisions with each other
            if self.collision_reward != 0:
                for a in self.world.policy_agents:
                    a.collision_rew[:] = 0

                for i, a in enumerate(self.world.agents):
                    for j, b in enumerate(self.world.agents):
                        if j <= i:
                            continue
                        collision = (
                            self.world.get_distance(a, b) <= self.min_collision_distance
                        )
                        if a.action_script is None:
                            a.collision_rew[collision] += self.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += self.collision_reward

        agent.dist_rew = torch.exp(-torch.linalg.vector_norm(agent.state.pos - self._target.state.pos, dim=-1)) * 0.01

        return agent.collision_rew + agent.dist_rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self._target.state.pos,
                agent.sensors[0].measure(),
            ],
            dim=-1,
        )

    def done(self):
        return self._done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "agent_collision_rew": agent.collision_rew,
            "agent_distance_rew": agent.dist_rew,
        }

        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        # First calculate the closest point to a circle of radius circle_radius given the current position
        circle_origin = torch.zeros(1, 2)
        circle_radius = 0.3
        current_pos = observation[:, :2]
        v = current_pos - circle_origin
        closest_point_on_circ = (
            circle_origin + v / torch.linalg.norm(v, dim=1).unsqueeze(1) * circle_radius
        )

        # calculate the normal vector of the vector from the origin of the circle to that closest point
        # on the circle. Adding this scaled normal vector to the other vector gives us a target point we
        # try to reach, thus resulting in a circular motion.
        closest_point_on_circ_normal = torch.stack(
            [closest_point_on_circ[:, Y], -closest_point_on_circ[:, X]], dim=1
        )
        closest_point_on_circ_normal /= torch.linalg.norm(
            closest_point_on_circ_normal, dim=1
        ).unsqueeze(1)
        closest_point_on_circ_normal *= 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents and obstcles within visibility range
        lidar = observation[:, 6:18]
        object_visible = torch.any(lidar < 0.1, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.1
        des_pos[object_visible] = des_pos_object[object_visible]

        action = torch.clamp(
            (des_pos - current_pos) * 10,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
