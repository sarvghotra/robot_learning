import numpy as np
import torch

from gym.spaces import Box
from hw1.roble.infrastructure import pytorch_util as ptu


class HRLWrapper(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, low_level_policy=None, **kwargs):
        self.base_env = base_env
        # TODO
        ## Load the policy \pi(a|s,g,\theta^{low}) you trained from Q4.
        ## Make sure the policy you load was trained with the same goal_frequency
        self.goal_frequency = kwargs['goal_frequency']

        self.gaussian_bounds = np.array(kwargs['gaussian_bounds'])
        low = self.gaussian_bounds[0] - 2*self.gaussian_bounds[1]
        high = self.gaussian_bounds[0] + 2*self.gaussian_bounds[1]
        self._action_space = Box(low=low, high=high)

        self._env_name = kwargs['env_name']

    def reset_step_counter(self):
        # TODO
        pass

    def seed(self, seed):
        np.random.seed(seed)
        self.base_env.seed(seed)

    def set_low_level_policy(self, low_level_policy, low_level_policy_path):
        self.low_level_policy = low_level_policy
        self.low_level_policy.load_state_dict(torch.load(low_level_policy_path, map_location=torch.device('cpu')))
        self.low_level_policy.to(ptu.device)
        return

    def reset(self):
        return self.base_env.reset()

    def success_fn(self, last_reward):
        if abs(last_reward) < abs(self._goal_reached_threshold):
            return True
        return False

    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO

        sub_goal = a # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        ob = self.base_env._get_obs()

        # TODO: uncomment the following line
        for i in range(self.goal_frequency):
            ob = self.create_state(ob, sub_goal)
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            ob = ptu.from_numpy(ob)
            ob = ob.to(ptu.device)
            a = self.low_level_policy(ob)
            ## Step the environment
            a = a.detach().cpu().numpy()
            ob, reward, done, info = self.base_env.step(a)

        info['reached_goal'] = self.success_fn(reward)
        return ob, reward, done, info

    def create_state(self, obs, goal):
        if self._env_name == 'reacher':
            obs[-3:] = goal
            return obs

        # return np.concatenate([obs, goal])
        raise "Not implemented"

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space

    @property
    def metadata(self):
        return self.base_env.metadata

    @property
    def unwrapped(self):
        return self.base_env
