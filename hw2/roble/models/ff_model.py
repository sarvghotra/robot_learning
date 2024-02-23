from torch import nn
import torch
from torch import optim
from .base_model import BaseModel
from hw2.roble.infrastructure.utils import *
from hw1.roble.infrastructure import pytorch_util as ptu


class FFModel(nn.Module, BaseModel):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **params):
        super(FFModel, self).__init__()

        self._delta_network = ptu.build_mlp(
            input_size=self._ob_dim + self._ac_dim,
            output_size=self._ob_dim,
            params = self._network
        )
        self._delta_network.to(ptu.device)
        self._optimizer = optim.Adam(
            self._delta_network.parameters(),
            self._learning_rate,
        )
        self._loss = nn.MSELoss()
        self._obs_mean = None
        self._obs_std = None
        self._acs_mean = None
        self._acs_std = None
        self._delta_mean = None
        self._delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self._obs_mean = ptu.from_numpy(obs_mean)
        self._obs_std = ptu.from_numpy(obs_std)
        self._acs_mean = ptu.from_numpy(acs_mean)
        self._acs_std = ptu.from_numpy(acs_std)
        self._delta_mean = ptu.from_numpy(delta_mean)
        self._delta_std = ptu.from_numpy(delta_std)

    def forward(
                self,
                obs_unnormalized,
                acs_unnormalized,
                obs_mean,
                obs_std,
                acs_mean,
                acs_std,
                delta_mean,
                delta_std,
        ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """

        obs_mean = convert_np_to_tensor_on_device(obs_mean)
        obs_std = convert_np_to_tensor_on_device(obs_std)
        acs_mean = convert_np_to_tensor_on_device(acs_mean)
        acs_std = convert_np_to_tensor_on_device(acs_std)
        delta_std = convert_np_to_tensor_on_device(delta_std)
        delta_mean = convert_np_to_tensor_on_device(delta_mean)

        # normalize input data to mean 0, std 1
        obs_unnormalized = convert_np_to_tensor_on_device(obs_unnormalized)
        acs_unnormalized = convert_np_to_tensor_on_device(acs_unnormalized)
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)

        # predicted change in obs
        acs_normalized = acs_normalized.to(torch.float32)
        obs_normalized = obs_normalized.to(torch.float32)
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self._delta_network(concatenated_input)

        delta_pred_unnormalized = unnormalize(delta_pred_normalized, delta_mean, delta_std)

        next_obs_pred = obs_unnormalized + delta_pred_unnormalized
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        # prediction = # TODO(Q1) get numpy array of the predicted next-states (s_t+1)
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        next_obs_pred, delta_pred_normalized = self.forward(obs,
                                                            acs,
                                                            data_statistics['obs_mean'],
                                                            data_statistics['obs_std'],
                                                            data_statistics['acs_mean'],
                                                            data_statistics['acs_std'],
                                                            data_statistics['delta_mean'],
                                                            data_statistics['delta_std'])
        next_obs_pred = next_obs_pred.detach().cpu().numpy()
        return next_obs_pred

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """
        delta_mean = data_statistics['delta_mean']
        delta_mean = convert_np_to_tensor_on_device(delta_mean)
        delta_std = data_statistics['delta_std']
        delta_std = convert_np_to_tensor_on_device(delta_std)

        _, delta_pred_normalized = self.forward(observations,
                                                actions,
                                                data_statistics['obs_mean'],
                                                data_statistics['obs_std'],
                                                data_statistics['acs_mean'],
                                                data_statistics['acs_std'],
                                                data_statistics['delta_mean'],
                                                data_statistics['delta_std'])

        next_observations = convert_np_to_tensor_on_device(next_observations)
        observations = convert_np_to_tensor_on_device(observations)
        target = normalize(next_observations - observations, delta_mean, delta_std) # TODO(Q1) compute the normalized target for the model.
        # # Hint: you should use `data_statistics['delta_mean']` and
        # # `data_statistics['delta_std']`, which keep track of the mean
        # # and standard deviation of the model.

        loss = self._loss(delta_pred_normalized, target) # TODO(Q1) compute the loss
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
