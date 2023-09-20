"""
JO-IPPO

Jointly Observable IPPO (JO-IPPO) provides the joint observations of all agents as input in addition to the agent
observation.

Configurations:
- Joint observations encoded by SAE to latent_dim (pre-trained/policy losses/reconstruction losses)
- Joint observations encoded by MLP to latent_dim (pre-trained/policy losses/reconstruction losses)
- Joint observations not encoded
"""

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from sae.model import AutoEncoder as PISA
import torch
from scenario_config import SCENARIO_CONFIG

POLICY_WIDTH=256 # Try 256? Try 1024 for melting pot
VALUE_WIDTH=256

class PolicyJOIPPO(TorchModelV2, torch.nn.Module):

    def __init__(self, observation_space, action_space, num_outputs, model_config, name, *args, **kwargs):

        # Call super class constructors
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_beta = False

        # Process keyword arguments
        self.scenario = kwargs["scenario"]
        self.task_agnostic = kwargs["task_agnostic"]
        self.task_specific = kwargs["task_specific"]
        self.train_specific = kwargs["train_specific"]
        self.pisa_dim = kwargs["pisa_dim"]
        self.no_comms = kwargs["no_comms"]
        self.pisa_path = kwargs["pisa_path"]
        self.n_agents = SCENARIO_CONFIG[self.scenario]["num_agents"]
        self.policy_width = kwargs["policy_width"]

        self.value_width = max(VALUE_WIDTH, self.policy_width)

        print("POLICY:", self.policy_width, "VALUE:", self.value_width)

        obs_size = observation_space.shape[0] // self.n_agents

        if self.task_agnostic or self.task_specific:

            # Load state dict.
            if self.task_agnostic:
                # Load pre-trained PISA
                self.pisa = PISA(
                    dim=self.pisa_dim,
                    hidden_dim=self.pisa_dim * self.n_agents
                ).to(device)
                self.pisa.load_state_dict(torch.load(
                    self.pisa_path,
                    map_location=torch.device(device)
                ))
            else:
                # Load entire model
                self.pisa = torch.load(
                    self.pisa_path,
                    map_location=torch.device(device)
                )

            # Freeze PISA
            for p in self.pisa.parameters():
                p.requires_grad = False
                p.detach_()
        
        if self.train_specific:
            # Construct randomly initialised PISA
            self.pisa = PISA(
                dim=self.pisa_dim,
                hidden_dim=self.pisa_dim * self.n_agents,
            ).to(device)

        # Perm-invariant state + own state + one-hot vector
        feature_dim = obs_size * self.n_agents + obs_size + self.n_agents

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=feature_dim,
                out_features=self.policy_width
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features=self.policy_width,
                out_features=self.policy_width,
            ),
            torch.nn.Tanh(),
        )
        for layer in self.policy_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        policy_last = torch.nn.Linear(
                in_features=self.policy_width,
                out_features=num_outputs // self.n_agents,  # Discrete: action_space[0].n
        )
        torch.nn.init.normal_(policy_last.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(policy_last.bias, mean=0.0, std=0.01)
        self.policy_head.add_module("policy_last", policy_last)

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=feature_dim,
                out_features=self.value_width
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features=self.value_width,
                out_features=self.value_width,
            ),
            torch.nn.Tanh(),
        )
        for layer in self.value_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
        value_last = torch.nn.Linear(
            in_features=self.value_width,
            out_features=1
        )
        torch.nn.init.normal_(value_last.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(value_last.bias, mean=0.0, std=0.01)
        self.value_head.add_module("value_last", value_last)

        self.current_value = None

    def forward(self, inputs, state, seq_lens):

        observation, batch, agent_features, n_batches = self.process_flat_obs(inputs["obs_flat"])

        x = observation

        if not self.no_comms:
            x = self.pisa.encoder(x, batch, n_batches=n_batches)
        else:
            x = x.reshape(n_batches, -1) # [batches, agents * obs_size]

        logits, values = [], []
        for i in range(self.n_agents):
            input_features = torch.cat((
                    torch.zeros_like(x) if self.no_comms else x,
                    agent_features[:, i],
                    torch.nn.functional.one_hot(
                        torch.tensor(i, device=x.device),
                        self.n_agents,
                    ).repeat(n_batches, 1)
                ), dim=1)
            values.append(
                self.value_head(input_features).squeeze(1)
            )
            logits.append(
                self.policy_head(input_features)
            )
        self.current_value = torch.stack(values, dim=1)
        logits = torch.cat(logits, dim=1)

        return logits, state

    def value_function(self):
        return self.current_value  # [batches, n_agents]

    def process_flat_obs(self, observation):

        # Standardize observations
        observation /= 5.0

        n_batches = observation.shape[0]
        observation = observation.reshape(n_batches, self.n_agents, -1)  # [batches, agents, obs_size]
        agent_features = observation.clone()

        observation = torch.flatten(observation, start_dim=0, end_dim=1)  # [batches * agents, obs_size]
        batch = torch.arange(n_batches, device=observation.device).repeat_interleave(self.n_agents)

        return observation, batch, agent_features, n_batches
