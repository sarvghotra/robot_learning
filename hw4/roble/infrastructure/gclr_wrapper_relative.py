
import numpy as np
from gym.spaces import Box
from roboverse.envs.widow250_eeposition import Widow250EEPositionEnv
# from hw2.roble.envs.reacher.reacher_env import Reacher7DOFEnv


class RelativeGoalConditionedEnv(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        # TODO
        self._env = base_env
        self.goal_reached_threshold = kwargs['goal_reached_threshold']
        self.uniform_bounds = np.array(kwargs['uniform_bounds'])
        self.gaussian_bounds = np.array(kwargs['gaussian_bounds'])
        self.goal_dist = kwargs['goal_dist']
        self.goal_rep = kwargs['goal_rep']
        self.agent_pos_ids = kwargs['goal_indicies']
        bounds = self.gaussian_bounds if self.goal_dist == 'normal' else self.uniform_bounds

        # if isinstance(self._env.unwrapped, Reacher7DOFEnv):
        # if isinstance(self._env.unwrapped, Widow250EEPositionEnv):
        #     low = np.concatenate((self._env.observation_space.low, np.array(bounds[0])),)
        #     high = np.concatenate((self._env.observation_space.high, np.array(bounds[1])),)
        #     self._observation_space = Box(low, high)
        # elif self.goal_rep == 'relative':
        low = np.concatenate((self._env.observation_space.low, np.array(bounds[0])),)
        high = np.concatenate((self._env.observation_space.high, np.array(bounds[1])),)
        self._observation_space = Box(low, high)
        # else:
        #     self._observation_space = self._env.observation_space

        if isinstance(self._env.unwrapped, Widow250EEPositionEnv):
            self._rel_goal_scale = 0.2
        else:
            self._rel_goal_scale = 0.3
        # self.observation_dim = self._env.observation_dim + len(kwargs['uniform_bounds'][0])

    # def _modify_obs_space(self, bounds):
    #     goal_dim = len(bounds[0])
    #     dtype = self._env.observation_space.dtype
    #     shape = (self._env.observation_space.shape[0] + goal_dim,)
    #     low = np.concatenate((self._env.observation_space.low, np.array(bounds[0])),)
    #     high = np.concatenate((self._env.observation_space.high, np.array(bounds[1])),)
    #     return Box(low, high, shape, dtype)

    def success_fn(self, last_reward):
        if abs(last_reward) < abs(self.goal_reached_threshold):
            return True
        return False

    def seed(self, seed):
        np.random.seed(seed)
        self._env.seed(seed)

    # old implementation
    # def reset(self):
    #     # Add code to generate a goal from a distribution
    #     ob = self._env.reset()
    #     goal = np.random.uniform(low=self.uniform_bounds[0],
    #                             high=self.uniform_bounds[1],)
    #                             # size=self.uniform_bounds[0].shape)
    #     # obs = np.concatenate((obs, goal),)
    #     self._env.model.site_pos[self._env.target_sid] = list(goal)
    #     ob = self._env._get_obs()
    #     return ob

    def _get_agent_pos(self):
        pass

    def _sample_goal(self, agent_pos=None):
        if self.goal_rep == 'relative':
            goal = np.random.normal(loc=agent_pos,
                                    scale=[self._rel_goal_scale, self._rel_goal_scale, self._rel_goal_scale])

        # elif self.goal_dist == 'normal':
        #     goal = np.random.normal(loc=self.gaussian_bounds[0],
        #                             scale=self.gaussian_bounds[1])
        # elif self.goal_dist == 'uniform':
        #     goal = np.random.uniform(low=self.uniform_bounds[0],
        #                             high=self.uniform_bounds[1],)
        #                             # size=self.uniform_bounds[0].shape)
        else:
            raise "Invalid goal distribution type"
        return goal

    # def _reacher_reset(self, goal_pos):
    #     _ = self._env.reset_model()

    #     self._env.model.site_pos[self._env.target_sid] = list(goal_pos)
    #     self._env.sim.forward()

    #     observation, _reward, done, _info = self._env.step(np.zeros(7))
    #     ob = self._env._get_obs()

    #     if self.goal_rep == 'relative':
    #         agent_pos = ob[self.agent_pos_ids]
    #         goal_pos = list(self._sample_goal(agent_pos=agent_pos))

    #         self._env.model.site_pos[self._env.target_sid] = list(goal_pos)
    #         self._env.sim.forward()

    #         observation, _reward, done, _info = self._env.step(np.zeros(7))
    #         ob = self._env._get_obs()
    #         ob = self.create_state(ob, goal_pos)

    #     return ob

    def _widowx_reset(self):
        ob = self._env.reset()
        return ob

    def reset(self):
        # if isinstance(self._env, Reacher7DOFEnv):
        if isinstance(self._env.unwrapped, Widow250EEPositionEnv):
            ob = self._widowx_reset()
        # else:
        #     ob = self._reacher_reset()

        agent_pos = ob[self.agent_pos_ids]
        self.goal = self._sample_goal(agent_pos)
        ob = self.create_state(ob, self.goal)
        return ob

    # def _reacher_step(self, a):
    #     ob, reward, done, info = self._env.step(a)
    #     info["reached_goal"] = self.success_fn(reward)

    #     if self.goal_rep == 'relative':
    #         goal_pos = np.array(self._env.model.site_pos[self._env.target_sid])
    #         ob = self.create_state(ob, goal_pos)

    #     return ob, reward, done, info


    def get_info(self, ee_pos, ee_target_pose, ee_distance_threshold):
        info = {}
        # ee_pos, ee_quat = bullet.get_link_state(self.robot_id, self.end_effector_index)
        target_coord = np.array(ee_target_pose)
        ee_coord = np.array(ee_pos)
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        if euclidean_dist_3d <= ee_distance_threshold:
            info['ee_pose_success'] = True
            info['target_coord'] = ee_target_pose
        else:
            info['ee_pose_success'] = False

        info['euclidean_distance'] = euclidean_dist_3d
        info['target_coord'] = ee_target_pose
        return info


    def _widowx_step(self, a):
        assert len(a.shape) == 1 or a.shape[0] == 1, "Batch mode not supported"
        action = a
        if len(a.shape) == 2:
            action = a[0]

        ob, _, _, info = self._env.step(action)

        agent_pos = ob[self.agent_pos_ids]
        target_pos = self.goal
        info = self.get_info(agent_pos, target_pos, self.goal_reached_threshold)
        reward = self._env.get_reward(info)
        done = self._env.done

        info["reached_goal"] = True if info['euclidean_distance'] < abs(self.goal_reached_threshold) else False
        ob = self.create_state(ob, self.goal)
        return ob, reward, done, info

    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        # if isinstance(self._env, Reacher7DOFEnv):
        if isinstance(self._env.unwrapped, Widow250EEPositionEnv):
            return self._widowx_step(a)
        # else:
        #     return self._reacher_step(a)

    def create_state(self, obs, goal):
        ## Add the goal to the state
        # TODO
        if self.goal_rep == 'relative':
            agent_pos = obs[self.agent_pos_ids]
            goal = goal - agent_pos
            return np.concatenate((obs, goal),)

    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def metadata(self):
        return self._env.metadata
    @property
    def unwrapped(self):
        return self._env
