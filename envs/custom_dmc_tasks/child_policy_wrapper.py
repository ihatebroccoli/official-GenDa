from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import gym
from gym import spaces


@dataclass
class EnvSpec:
    action_space: spaces.Space
    observation_space: spaces.Space


class ChildPolicyEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        cp_dict: dict,
        cp_action_range: float,
        cp_unit_length: bool,
        cp_multi_step: int,
        cp_num_truncate_obs: int,
        cp_omit_obs_idxs: Optional[Union[Sequence[int], np.ndarray]] = None,
    ):
        super().__init__(env)

        self.child_policy = cp_dict["policy"]
        if hasattr(self.child_policy, "eval"):
            self.child_policy.eval()

        self.cp_dim_action = int(cp_dict["dim_option"])
        self.cp_action_range = float(cp_action_range)
        self.cp_unit_length = bool(cp_unit_length)
        self.cp_multi_step = int(cp_multi_step)
        self.cp_num_truncate_obs = int(cp_num_truncate_obs)
        self.cp_omit_obs_idxs = None if cp_omit_obs_idxs is None else np.asarray(cp_omit_obs_idxs, dtype=int)
        self.cp_discrete = bool(cp_dict.get("discrete", False))

        self.observation_space = self.env.observation_space

        if self.cp_discrete:
            self.action_space = spaces.Discrete(n=self.cp_dim_action)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cp_dim_action,), dtype=np.float32)

        self.last_obs = None
        self.first_obs = None

    @property
    def spec(self) -> EnvSpec:
        return EnvSpec(action_space=self.action_space, observation_space=self.observation_space)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.last_obs = ret
        self.first_obs = ret
        return ret

        
    def _policy_to_action_np(self, last_obs, cp_action_arr) -> np.ndarray:
        
        pol = self.child_policy

        out = pol.get_action(last_obs, cp_action_arr, eval=True)
        return out

    def step(self, cp_action: Union[int, np.ndarray], debug: bool = False, **kwargs):
        cp_action_arr = np.array(cp_action, copy=True)
        cp_action_arr = cp_action_arr / (np.linalg.norm(cp_action_arr))
        sum_rewards = 0.0
        acc_infos = defaultdict(list)
        done_final = False
        for _ in range(self.cp_multi_step):
            action = self._policy_to_action_np(self.last_obs[:-self.cp_num_truncate_obs], cp_action_arr).astype(np.float32)
            lb = np.asarray(self.env.action_space.low, dtype=np.float32)
            ub = np.asarray(self.env.action_space.high, dtype=np.float32)
            action = lb + (action + 1.0) * (0.5 * (ub - lb))
            action = np.clip(action, lb, ub)

            # env step
            next_obs, reward, done, info = self.env.step(action, **kwargs)
            self.last_obs = next_obs

            sum_rewards += float(reward)
            for k, v in info.items():
                acc_infos[k].append(v)

            if info.get("done_internal", False):
                done_final = True
            if done:
                done_final = True
                break
        infos = {}
        for k, v in acc_infos.items():
            if debug:
                if k in ["coordinates", "next_coordinates", "ori", "next_ori"]:
                    infos[k] = np.concatenate(v).reshape(-1, v[0].shape[-1])
                elif k in ["ori_obs", "next_ori_obs"]:
                    infos[k] = v[-1]
                else:
                    if isinstance(v[0], np.ndarray):
                        infos[k] = np.array(v)
                    elif isinstance(v[0], tuple):
                        infos[k] = np.array([list(l) for l in v])
                    else:
                        infos[k] = sum(v)
            else:
                if k in ["coordinates", "ori"]:
                    infos[k] = v[0]
                else:
                    infos[k] = v[-1]

        return next_obs, sum_rewards, done_final, infos
