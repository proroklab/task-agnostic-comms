"""
HetIPPO
"""

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from scenario_config import SCENARIO_CONFIG

class PolicyHetIPPO(TorchModelV2, torch.nn.Module):

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

        # Policy net heads
        self.policy_heads = []
        for i in range(self.n_agents):
            policy_head = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=obs_size,
                    out_features=self.core_hidden_dim
                ),
                torch.nn.Tanh(),
                # torch.nn.Linear(
                #     in_features=self.core_hidden_dim,
                #     out_features=self.head_hidden_dim
                # ),
                # torch.nn.Tanh()
            )
            # Initialise final layer with zero mean and very small variance
            lin_out = torch.nn.Linear(
                in_features=self.core_hidden_dim,
                out_features=num_outputs // self.n_agents,  # Discrete: action_space[0].n
            )
            torch.nn.init.normal_(lin_out.weight, mean=0.0, std=0.03)
            torch.nn.init.normal_(lin_out.bias, mean=0.0, std=0.03)
            policy_head.add_module('lin_out', lin_out)
            self.policy_heads.append(policy_head)

        self.policy_heads = torch.nn.ModuleList(self.policy_heads)

        # Value head
        self.value_heads = []
        for i in range(self.n_agents):
            self.value_heads.append(
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=obs_size,
                        out_features=self.core_hidden_dim,
                    ),
                    torch.nn.Tanh(),
                    # torch.nn.Linear(
                    #     in_features=self.core_hidden_dim,
                    #     out_features=self.head_hidden_dim
                    # ),
                    # torch.nn.Tanh(),
                    torch.nn.Linear(in_features=self.core_hidden_dim, out_features=1)
                )
            )
        self.value_heads = torch.nn.ModuleList(self.value_heads)

        self.current_value = None

    def forward(self, inputs, state, seq_lens):

        observation = inputs["obs_flat"]  # [batches, agents * obs_size]
        n_batches = observation.shape[0]
        observation = observation.reshape(n_batches, self.n_agents, -1)  # [batches, agents, obs_size]
        agent_features = observation.clone()
        observation = torch.flatten(observation, start_dim=0, end_dim=1)  # [batches * agents, obs_size]
        observation = observation.reshape(n_batches, -1)  # [batches, agents * obs_size]
        x = observation

        values = []
        for i in range(self.n_agents):
            values.append(
                self.value_heads[i](agent_features[:, i].clone()).squeeze(1)
            )
        self.current_value = torch.stack(values, dim=1)

        logits = []
        for i in range(self.n_agents):
            logits.append(
                self.policy_heads[i](agent_features[:, i].clone())
            )
        logits = torch.cat(logits, dim=1)

        return logits, state

    def value_function(self):
        return self.current_value  # [batches, n_agents]
