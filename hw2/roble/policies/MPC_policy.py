# hw1 imports
from hw1.roble.policies.base_policy import BasePolicy

import numpy as np
import torch


class MPCPolicy(BasePolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 env,
                 dyn_models,
                 mpc_horizon,
                 mpc_num_action_sequences,
                 mpc_action_sampling_strategy='random',
                 **kwargs
                 ):
        super().__init__()

        # init vars
        self._data_statistics = None  # NOTE must be updated from elsewhere


        # action space
        self._ac_space = self._env.action_space
        self._low = self._ac_space.low
        self._high = self._ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert self._mpc_action_sampling_strategy in allowed_sampling, f"self._mpc_action_sampling_strategy must be one of the following: {allowed_sampling}"

        print(f"Using action sampling strategy: {self._mpc_action_sampling_strategy}")
        if self._mpc_action_sampling_strategy == 'cem':
            print(f"CEM params: alpha={self._cem_alpha}, "
                + f"num_elites={self._cem_num_elites}, iterations={self._cem_iterations}")

    def get_random_actions(self, num_sequences, horizon):
       act_dim = self._low.shape[0]
       return np.random.uniform(low=self._low, high=self._high,
						size=(num_sequences, horizon, act_dim))

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self._mpc_action_sampling_strategy == 'random' \
            or (self._mpc_action_sampling_strategy == 'cem' and obs is None):
            # TODO (Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self._ac_dim) in the range
            # [self._low, self._high]
            act_dim = self._low.shape[0]
            random_action_sequences = np.random.uniform(low=self._low, high=self._high,
                                                        size=(num_sequences, horizon, act_dim))
            return random_action_sequences
        elif self._mpc_action_sampling_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            ac_dim = self._ac_space.shape[0]
            gauss_mean = np.zeros((horizon, ac_dim))
            gauss_var = np.zeros((horizon, ac_dim))

            for i in range(self._cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                if i == 0:
                    candidate_action_sequences = self.get_random_actions(num_sequences, horizon)
                else:
                    candidate_action_sequences = np.empty((num_sequences, horizon, ac_dim))
                    for h in range(horizon):
                        candidate_action_sequences[:, h, :] = np.random.multivariate_normal(gauss_mean[h], np.diag(gauss_var[h]), num_sequences)

                # - Get the top `self._cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                ac_seqs_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
                # top_j = np.argpartition(a, -4)[-4:]
                top_j_ac_seqs_indices = ac_seqs_rewards.argsort()[-self._cem_num_elites:][::-1]
                top_elites = candidate_action_sequences[top_j_ac_seqs_indices]
                # - Update the elite mean and variance
                gauss_mean = np.mean(top_elites, axis=0)
                gauss_var = np.var(top_elites, axis=0)

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self._ac_dim)
            cem_action = gauss_mean
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self._mpc_action_sampling_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)

        bs, nb_acs, ac_dim = candidate_action_sequences.shape   # N, A, D
        # candidate_action_sequences = np.swapaxes(candidate_action_sequences, 0, 1)   # A, N, D
        ensemble_rewards = np.zeros((bs,))

        # obs = np.expand_dims(obs, axis=0)
        # obs = np.tile(obs, (bs, 1))

        for model in self._dyn_models:
            # reward = np.empty([nb_acs, bs])
            # next_obs_pred = obs

            # for i, ac in enumerate(candidate_action_sequences):
            #     reward[i] = self._env.get_reward(next_obs_pred, ac)[0]
            #     next_obs_pred = model.get_prediction(next_obs_pred, ac, self._data_statistics)

            # reward_sum = np.sum(reward, axis=0)
            ensemble_rewards += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)

        ensemble_rewards = ensemble_rewards / len(self._dyn_models)
        return ensemble_rewards

    def get_action(self, obs):
        if self._data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self._mpc_num_action_sequences, horizon=self._mpc_horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = np.argmax(predicted_rewards)  # TODO (Q2)
            action_to_take = candidate_action_sequences[best_action_sequence][0]  # TODO (Q2)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        bs, nb_acs, ac_dim = candidate_action_sequences.shape   # N, A, D
        candidate_action_sequences = np.swapaxes(candidate_action_sequences, 0, 1)   # A, N, D

        obs = np.expand_dims(obs, axis=0)
        obs = np.tile(obs, (bs, 1))

        reward = np.empty([nb_acs, bs])
        next_obs_pred = obs

        for i, ac in enumerate(candidate_action_sequences):
            reward[i] = self._env.get_reward(next_obs_pred, ac)[0]
            next_obs_pred = model.get_prediction(next_obs_pred, ac, self._data_statistics)

        sum_of_rewards = np.sum(reward, axis=0)  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self._env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self._mpc_horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
