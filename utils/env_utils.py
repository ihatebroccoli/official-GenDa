
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb


class ConsistentNormalizedEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        expected_action_scale: float = 1.0,
        flatten_obs: bool = True,
        normalize_obs: bool = True,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        super().__init__(env)
        self._normalize_obs = normalize_obs
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        obs_shape = env.observation_space.shape
        self._obs_mean = np.zeros(obs_shape) if mean is None else np.array(mean)
        self._obs_var = np.ones(obs_shape) if std is None else np.array(std)**2

        self._cur_obs = None

        if isinstance(env.action_space, gym.spaces.Box):
            act_shape = env.action_space.shape
            self.action_space = gym.spaces.Box(
                low=-self._expected_action_scale,
                high= self._expected_action_scale,
                shape=act_shape,
                dtype=env.action_space.dtype
            )
        else:
            self.action_space = env.action_space

        self.observation_space = env.observation_space

    def _apply_normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._cur_obs = obs

        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)

        if self._flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)

        return obs

    def step(self, action, **kwargs):
        if isinstance(self.env.action_space, gym.spaces.Box):
            lb, ub = self.env.action_space.low, self.env.action_space.high
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                # action in [-scale, scale] → [lb, ub]
                scaled = lb + (action + self._expected_action_scale) * 0.5 * (ub - lb) / self._expected_action_scale
                scaled_action = np.clip(scaled, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action, **kwargs)

        info['original_observations'] = self._cur_obs
        info['original_next_observations'] = next_obs
        self._cur_obs = next_obs

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._flatten_obs:
            next_obs = gym.spaces.utils.flatten(self.env.observation_space, next_obs)

        return next_obs, reward, done, info


consistent_normalize = ConsistentNormalizedEnv


from functools import partial
import functools


def get_normalizer_preset(normalizer_type, coord_only: bool = False):
    # Precomputed mean and std of the state dimensions from random rollouts
    if normalizer_type == 'off':
        normalizer_mean = np.array([0.])
        normalizer_std = np.array([1.])
    elif normalizer_type == 'fish_preset':
        normalizer_mean = np.array([-3.63591454e-04, -2.69135444e-04,  9.74342004e-02,  5.50009266e-04,
                                        8.04826189e-04,  4.84304159e-05,  1.18968146e-05,  5.61313370e-04,
                                    -5.86078599e-05, -3.32892089e-04,  2.09499001e-02, -4.55595241e-03,
                                        9.39693116e-03, -1.90452450e-01, -3.14784241e-05, -6.51188854e-05,
                                    -2.51018324e-04,  7.68324366e-04, -1.15501383e-04, -1.46385313e-04,
                                        5.53374811e-04,  8.46329816e-05,  4.35563647e-06, -6.76402426e-06,
                                    -1.00163614e-04, -3.23561617e-04, -2.40451016e-07])
        normalizer_std = np.array([0.08168116, 0.07838276, 0.07711183, 0.25003539, 0.16365467,
                                    0.20615534, 0.18493125, 0.1723247 , 0.18465988, 0.1717627 ,
                                    0.59358687, 0.19890836, 0.19533657, 0.21291593, 0.00929107,
                                    0.00911587, 0.00908252, 0.09377796, 0.18260293, 0.47944549,
                                    4.40128314, 1.26964046, 3.99235112, 2.27972832, 2.28131615,
                                    2.279327  , 2.2819299 ])
    elif normalizer_type == 'humanoid_numeric_preset':
        normalizer_mean = np.array([ 3.15720476e-02, -1.16678290e-01,  1.91259176e-01, -3.17427628e-02,
                                    -1.17611416e-01, -1.83843598e-02, -1.22785814e-01, -1.22782864e-01,
                                    -3.56559724e-01, -1.46745026e+00, -7.08044171e-02,  4.13410738e-02,
                                    -1.21270433e-01, -6.51777908e-02, -3.60923827e-01, -1.44340324e+00,
                                    -6.94715306e-02, -2.55798437e-02, -1.05887294e-01, -2.28047505e-01,
                                    -1.90649316e-01,  1.53801948e-01,  2.62089193e-01, -2.09952891e-01,
                                    2.00365856e-01,  1.54801637e-01,  2.43721798e-01,  2.02570558e-02,
                                    -2.30453201e-02,  1.08293444e-01, -7.82562554e-01,  1.65162206e-01,
                                    -2.32585430e-01,  1.98067185e-02, -3.19085605e-02, -1.25431314e-01,
                                    -7.77845681e-01,  2.56663591e-01,  9.66109112e-02,  4.79499474e-02,
                                    3.23509872e-02, -2.40706280e-02, -1.30069152e-01,  8.59995198e-04,
                                    -9.93560813e-03, -1.24652341e-01, -7.17373490e-02, -9.61070582e-02,
                                    1.92870386e-02, -4.35430231e-03,  1.63486615e-01,  2.88003590e-03,
                                    -3.58712412e-02, -5.07829804e-03, -6.06359579e-02,  1.31648168e-01,
                                    2.21012011e-01,  6.53506890e-02, -5.22161685e-02, -5.38052351e-04,
                                    -5.10843620e-02,  1.30086556e-01,  1.95036322e-01, -3.40666957e-02,
                                    -1.64407771e-02,  5.48741594e-02, -1.37457281e-01,  1.89375542e-02,
                                    -6.35815263e-02, -1.33654088e-01])

        normalizer_std = np.array([ 0.42375383,  0.41003507,  0.21795489,  0.3991815 ,  0.4309989 ,
                                    0.3655444 ,  0.19781657,  0.4866474 ,  0.6026855 ,  1.0167705 ,
                                    0.700499  ,  0.73529863,  0.19692637,  0.475169  ,  0.61110854,
                                    1.0214889 ,  0.70224077,  0.7389193 ,  0.8935857 ,  0.7411887 ,
                                    0.97295743,  0.89897346,  0.7503957 ,  0.97446793,  0.22246319,
                                    0.19580512,  0.21454859,  0.23759364,  0.2969462 ,  0.2842995 ,
                                    0.2680941 ,  0.19226612,  0.21071774,  0.23413058,  0.29081818,
                                    0.28204003,  0.26410076,  0.8160296 ,  0.45515534,  0.2222625 ,
                                    0.2830685 ,  0.28063276,  0.6756092 ,  0.4045586 ,  0.40480334,
                                    0.75882596,  1.8202374 ,  2.2940853 ,  3.930639  ,  4.00636   ,
                                    4.2322154 ,  3.3374217 ,  3.2422016 ,  3.9207973 ,  7.0018816 ,
                                    12.280487  , 18.196007  , 19.28512   ,  3.2385871 ,  3.907507  ,
                                    7.0685596 , 12.520766  , 18.416248  , 19.545387  ,  8.057647  ,
                                    8.014503  , 15.269329  ,  8.31628   ,  8.22643   , 15.4700165 ])
        
    elif normalizer_type == 'quadruped_numeric_preset':
        normalizer_mean = np.array([ 1.2926594e-02,  6.8294664e-04,  5.1880848e-01, -2.9323897e-03,
                                    -4.4155963e-02, -3.9512492e-04,  4.4107657e-02,  4.8024789e-03,
                                    -4.2075716e-02,  9.7075012e-04,  4.0712118e-02, -1.1407144e-03,
                                    -3.9830320e-02, -3.9062158e-03,  4.3337289e-02,  2.7766072e-03,
                                    -4.1873123e-02,  6.9446565e-04,  4.0715016e-02, -3.6056247e-03,
                                    -1.7475927e-02,  8.8627087e-03,  1.5154781e-02, -4.4189971e-03,
                                    -1.1149295e-02,  9.5117921e-03,  6.5249628e-03, -2.8679180e-03,
                                    -9.8636942e-03,  5.4717185e-03,  4.1380706e-03,  3.8914063e-03,
                                    -1.1366801e-02,  2.2631520e-03,  5.6867269e-03, -2.4342451e-03,
                                    4.6055622e-02, -5.7807483e-04,  3.9809393e-03,  5.0892647e-02,
                                    2.3020746e-05, -5.5658697e-05,  5.0390650e-02,  3.2176441e-03,
                                    2.0901891e-03,  5.0303865e-02, -4.9591024e-04,  3.6561303e-03,
                                    -5.4288516e-04, -3.0950930e-03,  9.8530459e-01, -1.6058672e-02,
                                    3.0558435e-02,  9.2399111e+00,  5.2531372e-04, -7.8803848e-04,
                                    -9.9757444e-03,  1.5459941e-01,  1.4609495e-02, -2.9514036e+00,
                                    1.7250814e-01,  8.3671734e-03, -2.9832187e+00,  1.6868359e-01,
                                    1.3475070e-02, -2.9704313e+00,  1.6148777e-01, -3.1038638e-02,
                                    -2.9791703e+00,  1.7033793e-02, -3.1642890e-01,  1.4181575e-02,
                                    -1.2002824e-03, -3.1976545e-01, -3.7135303e-03,  3.0434588e-03,
                                    -3.1302613e-01, -4.5317053e-03, -1.1025272e-02, -3.1540143e-01,
                                    6.9029969e-03])
        normalizer_std = np.array([0.3023603 , 0.29330307, 0.07384819, 0.23665681, 0.20943017,
                                    0.20353988, 0.23528564, 0.23745626, 0.21013649, 0.20392068,
                                    0.23300776, 0.23698007, 0.20954913, 0.20200251, 0.2343874 ,
                                    0.2404401 , 0.21150619, 0.20255736, 0.23462118, 3.1210127 ,
                                    2.2407587 , 2.178321  , 2.631944  , 3.0906157 , 2.2488174 ,
                                    2.187637  , 2.6223557 , 3.1115022 , 2.2491143 , 2.1889637 ,
                                    2.6214097 , 3.1303287 , 2.2517114 , 2.1769755 , 2.627414  ,
                                    0.25891203, 0.27259076, 0.20770869, 0.26031956, 0.27060086,
                                    0.20812337, 0.25984362, 0.2715593 , 0.2078318 , 0.26133493,
                                    0.2731239 , 0.20762575, 0.39851835, 0.4033406 , 0.5570268 ,
                                    0.01621253, 7.708779  , 7.706258  , 9.2287655 , 1.1484616 ,
                                    1.1391752 , 1.2852117 , 5.214581  , 5.092332  , 4.484689  ,
                                    5.2138906 , 5.0927672 , 4.4667234 , 5.2162323 , 5.0962114 ,
                                    4.4656696 , 5.22395   , 5.095213  , 4.478232  , 2.2909346 ,
                                    2.5137887 , 1.2594438 , 2.2943604 , 2.5136526 , 1.2674519 ,
                                    2.2858171 , 2.5160682 , 1.2659864 , 2.2887585 , 2.5219212 ,
                                    1.2605273 ])
        
    elif normalizer_type == 'dog_numeric_preset':
        normalizer_mean = np.array([ 0.045917, 0.029071, 0.148381, -0.009645, 0.000140, 0.000065, -0.002993, 0.000049, 0.000042, 0.006417, -0.000022, -0.000033, 0.023188, -0.000045, 0.052220, 0.032290, -0.149527, -0.845514, -0.621979, 0.042628, 0.055417, 0.030757, -0.147271, -0.844345, -0.624169, 0.044197, 0.158960, 0.000703, 0.150882, 0.000309, 0.066839, -0.000837, -0.005199, -0.000789, -0.033651, -0.001352, -0.072865, -0.000433, -0.088257, -0.000550, -0.080569, 0.000410, -0.058576, 0.000984, -0.027869, -0.000122, -0.005731, 0.130506, -0.003232, -0.001587, 0.113136, -0.002035, -0.000309, 0.068214, -0.002350, -0.000608, 0.039471, -0.003206, 0.112048, -0.118801, 0.075693, 0.001247, 0.006470, -0.078881, 0.054132, -0.551988, -0.801007, 0.113929, 0.073738, 0.001061, 0.005176, -0.079142, 0.059140, -0.553135, -0.801682, 0.115022, 0.001227, -0.000818, -0.000308, 0.001026, -0.000893, -0.000497, 0.000540, -0.000906, -0.000460, -0.000889, -0.001099, 0.003484, -0.001213, -0.002384, -0.054751, -0.063003, -0.039673, 0.000459, 0.001798, 0.001921, -0.057114, -0.065257, -0.030789, 0.010400, -0.001216, 0.010165, -0.001046, 0.004988, -0.000955, 0.000385, -0.000670, -0.001660, -0.000117, -0.004410, 0.000131, -0.005831, 0.000217, -0.005504, 0.000158, -0.003804, 0.000078, -0.001468, 0.000365, 0.000394, 0.005611, -0.002351, -0.000167, 0.004144, -0.001920, -0.000523, 0.002742, -0.001618, -0.000593, 0.002025, -0.001449, -0.014620, -0.013181, 0.001913, 0.009471, -0.005796, -0.001482, 0.019151, -0.087991, -0.069796, -0.064460, 0.004084, 0.008129, -0.006406, -0.004033, 0.020888, -0.092714, -0.071491, -0.077317, 0.148381, 0.137265, 0.183618, 0.006501, -0.055251, 0.035156, 0.009968, -0.280923, -0.073740, 0.007240, -0.251048, -0.027838, 0.000755, -0.010675, -4.997541, 0.096336, -4.892243, -0.059478, -0.001383, -0.035098, 0.000926, 0.048918, -0.002463, -1.177008, 0.434689, 0.294277, -1.129413, -0.444362, 0.282361, -0.524093, 0.301440, 1.711603, 0.084127, -0.575485, 0.633345, 6.659224, 5.395752, 6.536378, 6.611085, 0.000066, -0.000052, -0.000149, 0.000049, -0.001166, -0.000297, 0.000642, -0.000146, -0.002113, 0.000679, -0.000581, -0.000376, -0.000437, -0.001408, 0.001967, -0.000661, 0.000193, -0.000107, -0.001015, 0.000726, 0.000017, 0.000314, 0.000926, -0.000366, 0.000615, 0.000067, -0.000068, 0.000539, -0.000811, -0.000585, -0.000771, -0.000198, -0.000389, -0.000375, 0.001209, -0.000356, -0.000970, 0.000818,  ])
        normalizer_std = np.array([ 0.202298, 0.199142, 0.051089, 0.267830, 0.177026, 0.156462, 0.251692, 0.163101, 0.154818, 0.236062, 0.151610, 0.155828, 0.227271, 0.142396, 0.238966, 0.361581, 0.921083, 0.881134, 0.890257, 0.268378, 0.238600, 0.363433, 0.923616, 0.881438, 0.890983, 0.268294, 0.086828, 0.153617, 0.090081, 0.152753, 0.133975, 0.152554, 0.157077, 0.150236, 0.152445, 0.143912, 0.142004, 0.143983, 0.139165, 0.144696, 0.143746, 0.148854, 0.150460, 0.152149, 0.156459, 0.155166, 0.157074, 0.298809, 0.300671, 0.234667, 0.278998, 0.303648, 0.218138, 0.251147, 0.286442, 0.221766, 0.234905, 0.282825, 0.331915, 0.173662, 0.306648, 0.242677, 0.346523, 0.367186, 0.588164, 0.928880, 0.879134, 0.421059, 0.307064, 0.242889, 0.347350, 0.367860, 0.586975, 0.930083, 0.880200, 0.420721, 2.901964, 2.445458, 2.279958, 2.630831, 2.122685, 2.184736, 2.508882, 1.976473, 2.271062, 2.576279, 1.980116, 4.107576, 4.829603, 10.909613, 10.290721, 12.782244, 12.200521, 4.105733, 4.841422, 10.919464, 10.291527, 12.761479, 12.215178, 1.146125, 1.170165, 1.288088, 1.205128, 1.283105, 1.137387, 1.237655, 1.048316, 1.086686, 0.929970, 0.876360, 0.781059, 0.738513, 0.685724, 0.645836, 0.618445, 0.595113, 0.574426, 0.545138, 0.528486, 0.503126, 1.914340, 1.873113, 1.446251, 1.683639, 1.810938, 1.253738, 1.465178, 1.541482, 1.280792, 1.415678, 1.476747, 6.790944, 3.282952, 4.316095, 3.985780, 4.379154, 5.948256, 6.615934, 11.763342, 10.865802, 13.567736, 4.313362, 3.995099, 4.378122, 5.951144, 6.622414, 11.772739, 10.899546, 13.617145, 0.051089, 0.065734, 0.482173, 0.678149, 0.520292, 0.271387, 0.799458, 0.454827, 0.391885, 0.734221, 0.488617, 0.389624, 0.409698, 0.370831, 38.566872, 45.819469, 45.207260, 0.781352, 0.958028, 0.888570, 5.370014, 7.931351, 5.260320, 42.363937, 14.502815, 15.804536, 41.670574, 14.484072, 15.339181, 98.435928, 108.849258, 189.128677, 123.615234, 88.807365, 112.531075, 458.090515, 365.378326, 138.444534, 182.940536, 0.320002, 0.319408, 0.319634, 0.318921, 0.319711, 0.319792, 0.318933, 0.319365, 0.319936, 0.319199, 0.319024, 0.319400, 0.319467, 0.319662, 0.319869, 0.319588, 0.319586, 0.319412, 0.319168, 0.319238, 0.319341, 0.320145, 0.319580, 0.319456, 0.319834, 0.319257, 0.320404, 0.319491, 0.319314, 0.320116, 0.319936, 0.320196, 0.319817, 0.319935, 0.319661, 0.320009, 0.319928, 0.319440,  ])

    else:
        raise NotImplementedError

    return normalizer_mean, normalizer_std


def make_env(env_name, seed, use_encoder=False, max_path_length=200, normalizer_type='off', frame_stack=None, render_hw=100, camera_id=0):
    if env_name == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif env_name.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers_noarko import RenderWrapper
        assert use_encoder  # Only support pixel-based environments
        if env_name == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        elif env_name == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif env_name == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert use_encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, frame_stack)

    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    return env


class BinChecker:
    def __init__(self, bound_x=70, bound_y=70, bin_size=1):
        self.x_min, self.x_max = -bound_x, bound_x
        self.y_min, self.y_max = -bound_y, bound_y
        self.bin_size = bin_size

        self.n_x = int((self.x_max - self.x_min) / bin_size)
        self.n_y = int((self.y_max - self.y_min) / bin_size)
        self.grid = np.zeros((self.n_x, self.n_y), dtype=bool)

    def get_bin_indices(self, pts):
        arr = np.asarray(pts, dtype=np.float64)
        if pts.shape[-1] == 3:
            arr = arr[:, :2] 
        if arr.ndim == 1:
            arr = arr[None, :]
        x = arr[:, 0]
        y = arr[:, 1]

        ix = np.floor((x - self.x_min) / self.bin_size).astype(np.int32)
        iy = np.floor((y - self.y_min) / self.bin_size).astype(np.int32)

        ix = np.clip(ix, 0, self.n_x - 1)
        iy = np.clip(iy, 0, self.n_y - 1)

        if ix.size == 1:
            return int(ix[0]), int(iy[0])
        return ix, iy


    def mark_visited(self, pts):
        ix, iy = self.get_bin_indices(pts)
        self.grid[ix, iy] = True
        return self.grid
    

    def visualize_grid(self):
        plt.imshow(self.grid, origin='lower', interpolation='nearest')
        plt.title("Visited Bins")
        plt.xlabel("Bin X")
        plt.ylabel("Bin Y")
        image = wandb.Image(plt.gcf())
        plt.close()

        return image


    def eval_visualize_grid(self, pts, base_cmap="viridis", goals=None):
        fig, ax = plt.subplots()

        if pts.ndim == 3:                         # (N_episode, T, obs_dim)
            if pts.shape[-1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                cmap = plt.get_cmap('viridis')
                for i in range(pts.shape[0]):
                    ax.scatter(pts[i,:,0], pts[i,:,1], pts[i,:,2], c=cmap(i / pts.shape[0]), marker='o', s=5)
                image = wandb.Image(plt.gcf())
                plt.close()

                points = pts.reshape(-1, 3)
                bin_indices = np.floor(points / self.bin_size).astype(np.int64)
                unique_bins = np.unique(bin_indices, axis=0)

                return image, unique_bins.shape[0]

            n_eps = pts.shape[0]
            # -1 = background, 0..n_eps-1 = episode index
            label_grid = np.full((self.n_x, self.n_y), -1, dtype=int)

            for ep_idx, ep_pts in enumerate(pts):
                ix, iy = self.get_bin_indices(ep_pts)   
                label_grid[ix, iy] = ep_idx             

            episode_colors = [plt.get_cmap(base_cmap)(i / n_eps) for i in range(n_eps)]
            cmap = ListedColormap(["white", *episode_colors])

            im = ax.imshow(label_grid + 1,              
                            origin="lower",
                            interpolation="nearest",
                            vmin=0, vmax=n_eps,         
                            cmap=cmap)

            cbar = fig.colorbar(im, ticks=np.arange(1, n_eps + 1))
            cbar.ax.set_yticklabels([f"Ep {i}" for i in range(n_eps)])
            if goals is not None:
                for ep_idx, ep_pts in enumerate(pts):
                    # ep_pts: (T, 2) assumed → extract x,y
                    x = np.floor((ep_pts[:, 1] - self.x_min) / self.bin_size).astype(np.int32)
                    y = np.floor((ep_pts[:, 0] - self.y_min) / self.bin_size).astype(np.int32)

                    dx = np.diff(x)
                    dy = np.diff(y)

                    ax.quiver(
                        x[:-1], y[:-1],
                        dx, dy,
                        angles='xy', scale_units='xy', scale=1,
                        color=episode_colors[ep_idx],
                        width=0.002, alpha=0.9
                    )

        else:                                 
            ix, iy = self.get_bin_indices(pts)
            label_grid = np.zeros((self.n_x, self.n_y), dtype=int)
            label_grid[ix, iy] = 1
            cmap = ListedColormap(["white", "red"])
            ax.imshow(label_grid,
                    origin="lower",
                    interpolation="nearest",
                    cmap=cmap)

        if goals is not None:
            goals = np.array(goals)
            ax.scatter(
                (goals[:, 1] - self.x_min) / self.bin_size,
                (goals[:, 0] - self.y_min) / self.bin_size,
                marker='*', color='gold', label='Goals'
            )

        ax.set_title("Eval Visited Bins")
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        ax.set_aspect("equal")
        ax.grid(False)
        image = wandb.Image(plt.gcf())
        plt.close()
        return image, np.count_nonzero(label_grid != -1)



class RSNormWrapper:
    def __init__(self, env, pixel_dim=None, shape=(), dtype=np.float32):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata
        self.obs_rms = RunningMeanStd(shape=shape, dtype=dtype)
        self.epsilon = 1e-8
        self.pixel_dim = pixel_dim

    
    def reset(self, update=False):
        obs = self.env.reset()
        if update:
            if self.pixel_dim is not None:
                self.obs_rms.update(obs[None, self.pixel_dim:])
            else:
                self.obs_rms.update(obs[None, ...])
        return obs


    def step(self, action, render=False, update=False):
        
        if self.pixel_dim is not None:
            obs, reward, done, extra = self.env.step(action)
            if update:
                self.obs_rms.update(obs[None,self.pixel_dim:])
        else:
            obs, reward, done, extra = self.env.step(action)
            if update:
                self.obs_rms.update(obs[None, ...])
        
        return obs, reward, done, extra

    
    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        if self.pixel_dim is not None:
            return self.env.render(mode=mode, width=width, height=height)
        return self.env.render(mode, height, width, camera_id)


class RunningMeanStd:
    """
    This implementation is adapted from the Gymnasium API.
    ref: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/utils.py
    """
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def pad_episodes(eps_coors):
    lengths = [len(ep) for ep in eps_coors]
    max_len = max(lengths)
    padded = np.zeros((len(eps_coors), max_len, 2))
    for i, ep in enumerate(eps_coors):
        L = lengths[i]
        padded[i, :L] = ep
        padded[i, L:] = ep[-1]
    return padded
