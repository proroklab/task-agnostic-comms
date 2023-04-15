"""
IPPO
"""

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from scenario_config import SCENARIO_CONFIG

class PolicyIPPO(TorchModelV2, torch.nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name, *args, **kwargs):

        # Call super class constructors
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process keyword arguments
        scenario_name = kwargs.get("scenario_name")
        cwd = kwargs.get("cwd")
        self.core_hidden_dim = kwargs.get("core_hidden_dim")
        self.head_hidden_dim = kwargs.get("head_hidden_dim")
        self.n_agents = SCENARIO_CONFIG[scenario_name]["num_agents"]
        self.use_beta = kwargs.get("use_beta")

        obs_size = observation_space.shape[0] // self.n_agents

        self.core_network = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=obs_size,
                out_features=self.core_hidden_dim,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features=self.core_hidden_dim,
                out_features=self.core_hidden_dim,
            ),
            torch.nn.Tanh(),
        )

        for layer in self.core_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        # Initialise final layer with zero mean and very small variance
        self.policy_head = torch.nn.Linear(
            in_features=self.core_hidden_dim,
            out_features=num_outputs // self.n_agents,  # Discrete: action_space[0].n
        )
        torch.nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.policy_head.bias, mean=0.0, std=0.01)

        # Value head
        self.value_head = torch.nn.Linear(
            in_features=self.core_hidden_dim,
            out_features=1
        )
        torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.value_head.bias, mean=0.0, std=0.01)
        self.current_value = None

    def forward(self, inputs, state, seq_lens):

        observation = inputs["obs_flat"]  # [batches, agents * obs_size]
        n_batches = observation.shape[0]
        observation = observation.reshape(n_batches, self.n_agents, -1)  # [batches, agents, obs_size]
        agent_features = observation.clone()
        observation = torch.flatten(observation, start_dim=0, end_dim=1)  # [batches * agents, obs_size]
        observation = observation.reshape(n_batches, -1)  # [batches, agents * obs_size]
        x = observation

        logits, values = [], []
        for i in range(self.n_agents):
            p = self.core_network(agent_features[:, i].clone())
            values.append(
                self.value_head(p.clone()).squeeze(1)
            )
            logits.append(
                self.policy_head(p.clone())
            )
        self.current_value = torch.stack(values, dim=1)
        logits = torch.cat(logits, dim=1)

        return logits, state

    def value_function(self):
        return self.current_value  # [batches, n_agents]
