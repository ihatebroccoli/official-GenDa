import numpy as np
import gym
from gym import spaces


class NumericGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        max_path_length: int,
        goal_range: float,
        num_goal_steps: int,
        touch_end: bool = False,
        tight_goal_range: bool = True,
    ):
        super().__init__(env)

        self.max_path_length = int(max_path_length)
        self.goal_epsilon = 3.0 if goal_range >= 7.5 else 1.5
        self.goal_range = float(goal_range)
        self.num_goal_steps = int(num_goal_steps)
        self.env_touch_end = bool(touch_end)

        self.sum_of_rewards = 0.0
        self.tight_goal_range = tight_goal_range
        self.desired_reward = max(1.0, max_path_length / num_goal_steps)

        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,)).astype(np.float32)
        self.init_goal = self.cur_goal.copy()
        self.num_steps = 0
        self.goal_chase_steps = 0

        base_obs = self._peek_obs()
        obs_dim = int(base_obs.shape[0])
        low = -np.inf * np.ones(obs_dim + 2, dtype=np.float32)
        high = np.inf * np.ones(obs_dim + 2, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def _peek_obs(self) -> np.ndarray:
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) >= 1:
            obs = out[0]
        else:
            obs = out
        return np.asarray(obs, dtype=np.float32)

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.cur_goal], axis=-1)

    def reset(self, **kwargs):
        if self.tight_goal_range:
            theta = np.random.uniform(0, 2 * np.pi)
            x = self.goal_range * np.cos(theta)
            y = self.goal_range * np.sin(theta)
            self.cur_goal = np.array([x, y], dtype=np.float32)
        else:
            self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,)).astype(np.float32)
        self.init_goal = self.cur_goal.copy()
        self.num_steps = 0
        self.goal_chase_steps = 0
        self.sum_of_rewards = 0.0

        out = self.env.reset(**kwargs)
        return self._transform(out)

    def compute_reward(self, info: dict) -> float:
        self.num_steps += 1
        self.goal_chase_steps += 1
        xposafter, yposafter = info['next_coordinates']
        pos = np.array([xposafter, yposafter], dtype=np.float32)
        delta = np.linalg.norm(self.cur_goal - pos)

        if self.num_steps != 1 and delta <= self.goal_epsilon:
            reward = 1.0
        else:
            reward = 0.0 
            
        if (self.goal_chase_steps % self.num_goal_steps == 0) or (reward > 0.0):
            self.goal_chase_steps = 0
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
        
        return float(reward)

    def step(self, action, **kwargs):
        next_obs, _, done, info = self.env.step(action, **kwargs)  
        reward = self.compute_reward(info)
        self.sum_of_rewards += reward
        done = (self.num_steps == self.max_path_length) or (self.sum_of_rewards >= self.desired_reward)
        info['is_success'] = float(self.sum_of_rewards >= self.desired_reward)
        return self._transform(next_obs), reward, done, info
