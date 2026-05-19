import numpy as np
import gym
from gym import spaces

C = """
***************
*             *
*             *
***************
"""

C_eval = """
***************
*           G *
* P           *
***************
"""


L = """
*********
*****   *
*****   *
*****   *
*****   *
*       *
*       *
*       *
*********
"""

L_eval = """
*********
***** G *
*****   *
*****   *
*****   *
*       *
*       *
* P     *
*********
"""

H = """
*********
*       *
*       *
*****   *
*****   *
*****   *
*       *
*       *
*********
"""
H_eval = """
*********
* G     *
*       *
*****   *
*****   *
*****   *
*       *
* P     *
*********
"""

M = """
**************
*    **      *
*    **      *
*    **  **  *
*    **  **  *
***        ***
***        ***
*****  **    *
*****  **    *
*        **  *
*        **  *
*    **      *
*    **      *
**************
"""
M_eval = """
**************
*    **    G *
*    **      *
*    **  **  *
*    **  **  *
***        ***
***        ***
*****  **    *
*****  **    *
*        **  *
*        **  *
*    **      *
* P  **      *
**************
"""


def maze_str(name):
    if name == 'C':
        return C
    elif name == 'L':
        return L
    elif name == 'H':
        return H
    elif name == 'M':
        return M
    elif name == 'C_eval':
        return C_eval
    elif name == 'L_eval':
        return L_eval
    elif name == 'H_eval':
        return H_eval
    elif name == 'M_eval':
        return M_eval
    else:
        raise ValueError(f"Unknown maze name: {name}")


class NumericMazeWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        max_path_length: int,
        goal_epsilon: float,
        num_goal_steps: int,
    ):
        super().__init__(env)

        self.max_path_length = int(max_path_length)
        self.goal_epsilon = goal_epsilon
        self.goal_range = float(10.0)
        self.num_goal_steps = int(num_goal_steps)

        self.sum_of_rewards = 0.0
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
        self.init_goal = self.cur_goal.copy()
        self.num_steps = 0
        self.goal_chase_steps = 0
        self.sum_of_rewards = 0.0

        out = self.env.reset(**kwargs)
        self.cur_goal = np.array(self.env._env._env._env._env.task._goal_xy)
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
        
        return float(reward)

    def step(self, action, **kwargs):
        next_obs, _, done, info = self.env.step(action, **kwargs)  
        reward = self.compute_reward(info)
        self.sum_of_rewards += reward
        done = (self.num_steps == self.max_path_length) or (self.sum_of_rewards >= self.desired_reward)
        info['is_success'] = float(self.sum_of_rewards >= self.desired_reward)
        return self._transform(next_obs), reward, done, info
