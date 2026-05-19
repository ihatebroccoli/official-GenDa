import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class RenderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, hybrid: bool=False, color_set=None):
        super().__init__(env)
        self.hybrid = hybrid

        if getattr(env, 'physics', None) and env._domain == 'cheetah':
            tex_type = env.physics.model.tex_type
            for i, t in enumerate(tex_type):
                if t == 0:
                    h = env.physics.model.tex_height[i]
                    w = env.physics.model.tex_width[i]
                    adr = env.physics.model.tex_adr[i]
                    colors = [
                        (np.array(plt.cm.rainbow(np.clip((y / w - 0.5) * 4 + 0.5, 0, 1)))[:3] * 255).astype(np.uint8)
                        for y in range(w)
                    ]
                    for x in range(h):
                        for y in range(w):
                            idx = adr + (x * w + y) * 3
                            env.physics.model.tex_rgb[idx:idx+3] = colors[y]
  
        else:
            tex_type = env.physics.model.tex_type
            for i, t in enumerate(tex_type):
                if t == 0:
                    h = env.physics.model.tex_height[i]
                    w = env.physics.model.tex_width[i]
                    adr = env.physics.model.tex_adr[i]
                    for x in range(h):
                        for y in range(w):
                            idx = adr + (x * w + y) * 3
                            if color_set is not None:
                                env.physics.model.tex_rgb[idx:idx+3] = [
                                    color_set[0], color_set[1], color_set[2]
                                ]
                            else:
                                env.physics.model.tex_rgb[idx:idx+3] = [
                                    int(x / h * 255), int(y / w * 255), 128
                                ]
        if getattr(env, 'physics', None):
            env.physics.model.mat_texrepeat[:, :] = 1

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        if self.hybrid:
            self.ob_info = {
                'type': 'hybrid',
                'pixel_shape': (64, 64, 3),
                'state_shape': (self.env.observation_space.shape[0] - 64 * 64 * 3,),
            }
        else:
            self.ob_info = {
                'type': 'pixel',
                'pixel_shape': (64, 64, 3),
            }

    def _get_pixels(self):
        img = self.env.render(mode='rgb_array', width=64, height=64)
        return img.copy()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.hybrid:
            pixel = self._get_pixels().flatten()
            return np.concatenate([pixel, obs], axis=0)
        return self._get_pixels()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.hybrid:
            pixel = self._get_pixels().flatten()
            return np.concatenate([pixel, obs], axis=0), reward, done, info
        return self._get_pixels(), reward, done, info


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        assert hasattr(env, 'ob_info') and env.ob_info['type'] in ['pixel', 'hybrid']
        
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

        ph, pw, pc = env.ob_info['pixel_shape']
        self.orig_shape = (ph, pw, pc)
        self.flat_size = ph * pw * pc
        self.new_shape = (ph, pw, pc * num_frames)

        if env.ob_info['type'] == 'pixel':
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=self.new_shape, dtype=np.float32
            )
            self.ob_info = {
                'type': 'pixel',
                'pixel_shape': self.new_shape,
            }
        else:  
            state_dim = np.prod(env.ob_info['state_shape'])
            obs_dim = np.prod(self.new_shape) + state_dim
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.ob_info = {
                'type': 'hybrid',
                'pixel_shape': self.new_shape,
                'state_shape': env.ob_info['state_shape'],
            }

    def _extract_pixels(self, obs: np.ndarray) -> np.ndarray:
        return obs[:self.flat_size].reshape(self.orig_shape)

    def _pack(self, pixel_stack: np.ndarray, rest: np.ndarray = None) -> np.ndarray:
        flat_pixels = pixel_stack.flatten()
        if rest is None:
            return flat_pixels
        return np.concatenate([flat_pixels, rest], axis=0)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        px = self._extract_pixels(obs)
        for _ in range(self.num_frames):
            self.frames.append(px)
        stacked = np.concatenate(list(self.frames), axis=2)
        rest = None
        if self.ob_info['type'] == 'hybrid':
            rest = obs[self.flat_size:]
        return self._pack(stacked, rest)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        px = self._extract_pixels(obs)
        self.frames.append(px)
        stacked = np.concatenate(list(self.frames), axis=2)
        rest = None
        if self.ob_info['type'] == 'hybrid':
            rest = obs[self.flat_size:]
        return self._pack(stacked, rest), reward, done, info
