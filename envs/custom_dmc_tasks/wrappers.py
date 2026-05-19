import numpy as np
from dm_env import specs
from gym import core, spaces
from collections import OrderedDict
import textwrap
import xml.etree.ElementTree as ET
import math
def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        if type(s) == OrderedDict:
            shapes = [int(np.prod(v.shape)) for v in s.values()]
            bounds = np.inf * np.ones((sum(shapes),), dtype=np.float32)
            return -bounds, bounds

        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


def _clean_maze_str(maze: str) -> list[str]:
    maze = textwrap.dedent(maze).strip("\n")
    lines = maze.splitlines()
    if not lines:
        raise ValueError("maze is empty")
    w = max(len(x) for x in lines)
    return [x.ljust(w) for x in lines]

def _find_all_tokens(lines: list[str], token: str) -> list[tuple[int, int]]:
    out = []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == token:
                out.append((r, c))
    return out

def _maze_to_world_xy_origin_left1_bottom1(
    row: int,
    col: int,
    H: int,
    W: int,          # 
    cell_size: float,
    origin_col: int = 1,
    origin_row_from_bottom: int = 1,  # bottom+1
) -> tuple[float, float]:
    """
    ✅ (left+1, bottom+1) 셀을 (0,0)으로 하는 좌표계.
    - origin_col=1  => col=1이 x=0
    - origin_row_from_bottom=1 => row=H-2가 y=0
    """
    origin_row = (H - 1) - origin_row_from_bottom 
    x = (col - origin_col) * cell_size
    y = (origin_row - row) * cell_size
    return x, y

def inject_maze_walls_into_xml(
    xml_string: str,
    maze: str,
    *,
    wall_token: str = "*",
    spawn_token: str = "P",
    goal_token: str = "G",
    cell_size: float = 1.0,
    wall_height: float = 1.5,
    wall_thickness_scale: float = 1.0,
    wall_rgba=(0.4, 0.4, 0.4, 1.0),
    add_ground_plane_if_missing: bool = False,

   
    randomize_spawn: bool = False,              
    randomize_goals: bool = False,              
    allow_random_spawn_if_missing: bool = True, 
    allow_random_goals_if_missing: bool = True, 
    num_random_goals: int = 1,
    rng: np.random.RandomState | None = None,
    seed: int | None = None,

    add_markers: bool = True,
    marker_size: float = 0.16,
    spawn_marker_rgba=(0.2, 0.8, 0.2, 1.0),
    goal_marker_rgba=(0.9, 0.2, 0.2, 1.0),

    add_top_camera: bool = True,
    top_camera_name: str = "maze_top",
    top_camera_fovy_deg: float = 55.0,
    top_camera_margin_cells: float = 1.0,
    top_camera_z_min: float = 2.0,
    top_camera_xyaxes: str = "1 0 0  0 1 0",
):
    """
      new_xml_string: str
      spawn_xy: (x,y)              
      goal_xys: list[(x,y)]        
    """
    if rng is None:
        rng = np.random.RandomState(seed)

    lines = _clean_maze_str(maze)
    H, W = len(lines), len(lines[0])

    open_cells = [(r, c) for r in range(H) for c in range(W) if lines[r][c] != wall_token]
    if not open_cells:
        raise ValueError("There is no feasible cell in the maze.")

    P_list = _find_all_tokens(lines, spawn_token)
    G_list = _find_all_tokens(lines, goal_token)

    if randomize_spawn:
        spawn_rc = open_cells[rng.randint(len(open_cells))]
    else:
        if P_list:
            spawn_rc = P_list[0]
        else:
            if allow_random_spawn_if_missing:
                spawn_rc = open_cells[rng.randint(len(open_cells))]
                randomize_spawn = True
            else:
                raise ValueError(f"There is no {spawn_token!r} in the maze and allow_random_spawn_if_missing=False.")

    # --- Goals 선택 (grid 좌표) ---
    if randomize_goals:
        candidates = [rc for rc in open_cells if rc != spawn_rc]
        if len(candidates) < num_random_goals:
            raise ValueError("There are not enough goal candidates. (Too few open cells or num_random_goals is too large)")
        idx = rng.choice(len(candidates), size=num_random_goals, replace=False)
        goals_rc = [candidates[i] for i in idx]
    else:
        if G_list:
            goals_rc = [rc for rc in G_list if rc != spawn_rc]
        else:
            if allow_random_goals_if_missing:
                candidates = [rc for rc in open_cells if rc != spawn_rc]
                if len(candidates) < num_random_goals:
                    raise ValueError("There are not enough goal candidates. (Too few open cells or num_random_goals is too large)")
                idx = rng.choice(len(candidates), size=num_random_goals, replace=False)
                randomize_goals = True
                goals_rc = [candidates[i] for i in idx]
            else:
                goals_rc = []

    # --- XML 파싱 ---
    root = ET.fromstring(xml_string)
    size = ET.SubElement(root, "size")
    size.set("nconmax", str(200))
    worldbody = root.find("worldbody")


    if worldbody is None:
        raise RuntimeError("MJCF XML does not contain <worldbody>.")

    # (선택) ground plane
    if add_ground_plane_if_missing:
        has_plane = any((g.tag == "geom" and g.get("type") == "plane") for g in worldbody.findall("geom"))
        if not has_plane:
            ET.SubElement(
                worldbody, "geom",
                dict(
                    name="ground", type="plane",
                    size="50 50 0.1",
                    rgba="0.9 0.9 0.9 1",
                    contype="1", conaffinity="1"
                )
            )

    # --- Camera/Bounding: Calculate overall bbox based on left+1,bottom+1 coordinate system ---
    xs, ys = [], []
    for r in range(H):
        for c in range(W):
            x, y = _maze_to_world_xy_origin_left1_bottom1(r, c, H, W, cell_size)
            xs.append(x); ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if add_top_camera:
        for cam in list(worldbody.findall("camera")):
            if cam.get("name") == top_camera_name:
                worldbody.remove(cam)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        half_w = (x_max - x_min) / 2.0 + top_camera_margin_cells * cell_size
        half_h = (y_max - y_min) / 2.0 + top_camera_margin_cells * cell_size
        half_diag = math.sqrt(half_w**2 + half_h**2)

        fovy_rad = math.radians(top_camera_fovy_deg)
        z = max(top_camera_z_min, half_diag / max(1e-6, math.tan(fovy_rad / 2.0)))

        ET.SubElement(
            worldbody, "camera",
            dict(
                name=top_camera_name,
                pos=f"{cx:.5f} {cy:.5f} {z:.5f}",
                xyaxes=top_camera_xyaxes,
                fovy=str(float(top_camera_fovy_deg)),
            )
        )

    # --- Wall geom addition ---
    wall_half_x = (cell_size * wall_thickness_scale) / 2.0
    wall_half_y = (cell_size * wall_thickness_scale) / 2.0
    wall_half_z = wall_height / 2.0
    wall_rgba_str = " ".join(str(float(x)) for x in wall_rgba)

    for r in range(H):
        for c in range(W):
            if lines[r][c] != wall_token:
                continue
            x, y = _maze_to_world_xy_origin_left1_bottom1(r, c, H, W, cell_size)
            ET.SubElement(
                worldbody, "geom",
                dict(
                    name=f"maze_wall_r{r}_c{c}",
                    type="box",
                    pos=f"{x:.5f} {y:.5f} {wall_half_z:.5f}",
                    size=f"{wall_half_x:.5f} {wall_half_y:.5f} {wall_half_z:.5f}",
                    rgba=wall_rgba_str,
                    contype="1",
                    conaffinity="1",
                    friction="1 0.5 0.5",
                )
            )

    # --- spawn/goal world coordinates (left+1,bottom+1 origin coordinate system) ---
    sr, sc = spawn_rc
    spawn_xy = _maze_to_world_xy_origin_left1_bottom1(sr, sc, H, W, cell_size)

    goal_xys = []
    for gr, gc in goals_rc:
        goal_xys.append(_maze_to_world_xy_origin_left1_bottom1(gr, gc, H, W, cell_size))

    # --- (Optional) Marker addition (for visualization) ---
    if add_markers:
        sm_rgba = " ".join(str(float(x)) for x in spawn_marker_rgba)
        gm_rgba = " ".join(str(float(x)) for x in goal_marker_rgba)

        sx, sy = spawn_xy
        ET.SubElement(
            worldbody, "geom",
            dict(
                name="maze_spawn_marker",
                type="sphere",
                pos=f"{sx:.5f} {sy:.5f} {marker_size:.5f}",
                size=f"{marker_size:.5f}",
                rgba=sm_rgba,
                contype="0",
                conaffinity="0",
            )
        )

        for i, (gx, gy) in enumerate(goal_xys):
            ET.SubElement(
                worldbody, "geom",
                dict(
                    name=f"maze_goal_marker_{i}",
                    type="sphere",
                    pos=f"{gx:.5f} {gy:.5f} {marker_size:.5f}",
                    size=f"{marker_size:.5f}",
                    rgba=gm_rgba,
                    contype="0",
                    conaffinity="0",
                )
            )

    new_xml_string = ET.tostring(root, encoding="unicode")
    return new_xml_string, (float(spawn_xy[0]), float(spawn_xy[1])), [(float(x), float(y)) for (x, y) in goal_xys], randomize_spawn, randomize_goals, open_cells, H, W, cell_size


class DMCGymWrapper(core.Env):
    def __init__(
            self,
            env,
            from_pixels=False,
            height=84,
            width=84,
            channels_first=True,
            domain='',
    ):
        self._env = env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._root_name = 'torso'
        if domain in ['humanoid_CMU']:
            self._root_name = 'root_geom'

        if domain == 'quadruped':
            self._camera_id = 2
        elif domain == 'quadmaze':
            self._camera_id = 1
            domain = 'quadruped'
        else:
            self._camera_id = 0
        
        self._channels_first = channels_first
        self._frame_skip = 1
        self._domain = domain

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                [self._env.observation_spec()],
                np.float64
            )

        self._state_space = _spec_to_box(
            [self._env.observation_spec()],
            np.float64
        )

        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            if type(time_step.observation) == OrderedDict:
                obs = _flatten_obs(time_step.observation)
                xyz = self.physics.named.data.geom_xpos[[self._root_name], ['x', 'y', 'z']].copy()
                obs = np.concatenate([xyz, obs], axis=0)
            else:
                obs = time_step.observation
            

        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action, render=False):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}
        xyz_before = self.physics.named.data.geom_xpos[[self._root_name], ['x', 'y', 'z']].copy()
        obsbefore = self.physics.get_state()

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        xyz_after = self.physics.named.data.geom_xpos[[self._root_name], ['x', 'y', 'z']].copy()

        obs = self._get_obs(time_step)
        self.current_state = time_step.observation
        obsafter = self.physics.get_state()
        extra['discount'] = time_step.discount

        if render:
            extra['render'] = self.render(mode='rgb_array', width=64, height=64).transpose(2, 0, 1)
    
        if self._domain in ['cheetah']:
            extra['coordinates'] = np.array([xyz_before[0], 0.])
            extra['next_coordinates'] = np.array([xyz_after[0], 0.])
        elif self._domain in ['quadruped', 'humanoid', 'dog', 'humanoid_CMU']:
            extra['coordinates'] = np.array([xyz_before[0], xyz_before[1]])
            extra['next_coordinates'] = np.array([xyz_after[0], xyz_after[1]])
        elif self._domain in ['fish']:
            extra['coordinates'] = np.array(xyz_before)
            extra['next_coordinates'] = np.array(xyz_after)
        extra['ori_obs'] = obsbefore
        extra['next_ori_obs'] = obsafter

        return obs, reward, done, extra

    def calc_eval_metrics(self, trajectories, is_option_trajectories=False):
        return dict()

    def compute_reward(self, ob, next_ob, action=None):
        xposbefore = ob[:, 0]
        xposafter = next_ob[:, 0]

        reward = (xposafter - xposbefore) / self.dt
        done = np.zeros_like(reward)

        return reward, done

    def reset(self):
        time_step = self._env.reset()
        self.current_state = time_step.observation
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def plot_trajectory(self, trajectory, color, ax):
        if self._domain in ['cheetah']:
            trajectory = trajectory.copy()
            from matplotlib.collections import LineCollection
            linewidths = np.linspace(0.2, 1.2, len(trajectory))
            points = np.reshape(trajectory, (-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=linewidths, color=color)
            ax.add_collection(lc)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            coordinates_trajectories.append(np.concatenate([
                trajectory['env_infos']['coordinates'],
                [trajectory['env_infos']['next_coordinates'][-1]]
            ]))
        if self._domain in ['cheetah']:
            for i, traj in enumerate(coordinates_trajectories):
                traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = {}

        coord_dim = 2 if self._domain in ['quadruped', 'humanoid', 'dog'] else 1

        coords = []
        for traj in trajectories:
            traj1 = traj['env_infos']['coordinates'][:, :coord_dim]
            traj2 = traj['env_infos']['next_coordinates'][-1:, :coord_dim]
            coords.append(traj1)
            coords.append(traj2)
        coords = np.concatenate(coords, axis=0)
        uniq_coords = np.unique(np.floor(coords), axis=0)
        eval_metrics.update({
            'MjNumTrajs': len(trajectories),
            'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
            'MjNumCoords': len(coords),
            'MjNumUniqueCoords': len(uniq_coords),
        })

        return eval_metrics
