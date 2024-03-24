import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy
from hw3.roble.utils.policy_utils import SquashedNormal


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network,
                                      is_critic=self._is_critic)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.15
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def scale_action(self, raw_action):
        # scaled_action = self._action_scale * raw_action
        # return scaled_action
        # FIXME
        return raw_action

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO:
        ## Provide the logic to produce an action from the policy
        # FIXME: double check if we need don't need backprop here
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        t_obs = torch.from_numpy(obs).float().to(ptu.device)
        with torch.no_grad():
            action = self.forward(t_obs)

        if isinstance(action, distributions.distribution.Distribution):
            action = action.rsample()

        # action = self.scale_action(action)
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                ##  TODO output for a deterministic policy
                # action_distribution = TODO
                action_distribution = self._mean_net(observation)
                # FIXME
                action_distribution = self._action_scale * action_distribution
            else:

                ##  TODO output for a stochastic policy
                # action_distribution = TODO
                # FIXME: double check it with another implementation
                # z = self._norm_dist.sample().to(ptu.device)
                std = torch.exp(self._logstd)
                # action_distribution = distributions.Normal(self._mean_net(observation), std)
                action_distribution = SquashedNormal(self._mean_net(observation), std)
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):

        self._optimizer.zero_grad()
        # TODO: update the policy and return the loss
        t_actions = torch.from_numpy(actions).to(ptu.device)
        t_observations = torch.from_numpy(observations).to(ptu.device)
        pred_actions = self.forward(t_observations)
        loss = self._loss(pred_actions, t_actions)
        loss.backward()
        self._optimizer.step()
        # FIXME: remove print
        # print("loss: ", ptu.to_numpy(loss))
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):


        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        # FIXME: guess bases created the input
        self._optimizer.zero_grad()
        # TODO: update the policy and return the loss
        t_actions = torch.from_numpy(actions).float().to(ptu.device)
        obs_nxt_obs = np.concatenate((observations, next_observations), axis=1)
        t_obs_nxt_obs = torch.from_numpy(obs_nxt_obs).float().to(ptu.device)

        pred_actions = self.forward(t_obs_nxt_obs)
        loss = self._loss(pred_actions, t_actions)
        loss.backward()
        self._optimizer.step()
        # FIXME: remove print
        # print("IDM loss: ", ptu.to_numpy(loss))
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }