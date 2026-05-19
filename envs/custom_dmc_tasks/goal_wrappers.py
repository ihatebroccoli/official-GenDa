from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt  
from gym import spaces


class GoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        max_path_length,
        goal_range,
        num_goal_steps,
        touch_end: bool = False,
        tight_goal_range: bool = True,
        randstart: bool = False,
    ):
        super().__init__(env)

        self.max_path_length = max_path_length

        self.goal_epsilon = 3.0 if goal_range >= 7.5 else 1.5
        self.tight_goal_range = tight_goal_range
        self.goal_range = goal_range
        self.num_goal_steps = num_goal_steps
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.num_steps = 0
        self.env_touch_end = bool(touch_end)
        self.randstart = randstart

        obs_dim = 64 * 64 * 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.ob_info = dict(
            type="hybrid",
            pixel_shape=(64, 64, 3),
            state_shape=2,
        )

    def _transform(self, obs):
        pixels = self.env.render(mode="rgb_array", width=64, height=64).copy()
        pixels = pixels.flatten()
        return np.concatenate([pixels, self.cur_goal], axis=-1)

    def reset(self, **kwargs):
        if self.tight_goal_range:
            theta = np.random.uniform(0, 2 * np.pi)
            x = self.goal_range * np.cos(theta)
            y = self.goal_range * np.sin(theta)
            self.cur_goal = np.array([x, y], dtype=np.float32)
        else:
            self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,)).astype(np.float32)
        
        obs = self.env.reset(**kwargs)
        if self.randstart:
            rand_pos = np.random.uniform(-self.goal_range, self.goal_range, size=(2,))
            self.physics.named.data.qpos['root'][:2] = rand_pos
            self.physics.forward()
            obs, _,_,_ = self.env.step(np.zeros(self.env.action_space.shape))

        self.num_steps = 0
        return self._transform(obs)

    def compute_reward(self, info):
        self.num_steps += 1
        xposafter, yposafter = info["next_coordinates"]
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter, yposafter]))
        if self.num_steps != 1 and delta <= self.goal_epsilon:
            reward = 1.0
        else:
            reward = 0.0

        if self.num_steps % self.num_goal_steps == 0:
            if self.tight_goal_range:
                theta = np.random.uniform(0, 2 * np.pi)
                x = self.goal_range * np.cos(theta)
                y = self.goal_range * np.sin(theta)
                self.cur_goal = np.array([x + xposafter, y + yposafter], dtype=np.float32)
            else:
                self.cur_goal = np.array([
                    np.random.uniform(xposafter - self.goal_range, xposafter + self.goal_range),
                    np.random.uniform(yposafter - self.goal_range, yposafter + self.goal_range),
                ], dtype=np.float32)

        return reward

    def step(self, action, **kwargs):
        next_obs, _, done, info = self.env.step(action, **kwargs)
        reward = self.compute_reward(info)
        done = (self.num_steps == self.max_path_length) or (reward >= 1.0 and self.env_touch_end)
        info['is_success'] = float(reward >= 1.0)
        return self._transform(next_obs), reward, done, info
