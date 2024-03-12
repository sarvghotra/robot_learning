import abc
import itertools
import numpy as np
import torch
import hw1.roble.util.class_util as classu

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy
from hw1.roble.policies.MLP_policy import MLPPolicy
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from torch.nn import utils

class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)

    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        observations = ptu.from_numpy(observations)
        actions = self.forward(observations)
        # actions = self.scale_action(actions)

        ## Hint: do not update the parameters for q_fun in the loss
        # with torch.no_grad():
        loss = q_fun._q_net(observations, actions)
        loss = -1.0 * loss.mean()   # acent

        self._optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.parameters(), self._grad_norm_clipping)
        self._optimizer.step()

        return {"Loss": loss.item()}

class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        t_obs = torch.from_numpy(obs).float().to(ptu.device)
        with torch.no_grad():
            action = self.forward(t_obs)

        if isinstance(action, distributions.distribution.Distribution):
            action = action.rsample()

        return ptu.to_numpy(action)


    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff

        observations = ptu.from_numpy(observations)
        action_dist = self.forward(observations)
        actions = action_dist.rsample()

        q_loss = torch.minimum(q_fun._q_net1(observations, actions), q_fun._q_net2(observations, actions))
        loss = q_loss - self.entropy_coeff * action_dist.log_prob(actions)
        loss = -1.0 * loss.mean()   # decent by default in Pytorch

        self._optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.parameters(), self._grad_norm_clipping)
        self._optimizer.step()


        return {"Loss": loss.item()}

#####################################################