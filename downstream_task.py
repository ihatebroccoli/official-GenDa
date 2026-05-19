from envs.custom_dmc_tasks import dmc
from utils.replay_buffer import HEREpisodeReplayBuffer

import argparse
import datetime, time
from typing import Dict, Tuple, Any
from utils.env_utils import BinChecker, ConsistentNormalizedEnv, get_normalizer_preset, pad_episodes
import wandb
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        description="RL Training Hyperparameters"
    )
    parser.add_argument('--device',      type=str,   default='0',   help='cuda visible devices')

    parser.add_argument('--obs_dim',      type=int,   default=70,   help='Observation space dimension')
    parser.add_argument('--action_dim',   type=int,   default=21,   help='Action space dimension')
    parser.add_argument('--option_dim',   type=int,   default=2,    help='Option space dimension')
    parser.add_argument('--phi_dims',     type=int,   default=2,    help='Phi network output dimension')
    parser.add_argument('--pixel_shape',     type=int,   default=2)
    parser.add_argument('--pixel_dim',     type=int,   default=2)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    parser.add_argument('--local_latent_dim',     type=int,   default=27,    help='Phi network output dimension')
    parser.add_argument('--z_relabeling', action='store_true', help='relabeling options with final - initi observations')
    parser.add_argument('--csf_weight_low_bound',     type=float,   default=0.1,    help='Phi network output dimension')
    parser.add_argument('--after_c_step',     type=int,   default=10,    help='Phi network output dimension')

    # hierarchical
    parser.add_argument('--high_level_action_freq',     type=int,   default=25,    help='High level action frequency')

    # agent
    parser.add_argument('--n_critics',   type=int,   default=2,    help='Number of critics')
    parser.add_argument('--algo',     type=str,   default='METRA', help='algorithm name')
    parser.add_argument('--with_local_state', action='store_true')
    parser.add_argument('--toggle_actor', action='store_true')
    parser.add_argument('--pseudo_local_encoder', action='store_true')
    parser.add_argument('--noresidual_local_encoder', action='store_true')
    parser.add_argument('--forward_dynamics', action='store_true', help='use forward dynamics model')
    parser.add_argument('--use_prior', action='store_true', help='use prior encoder, default for LFSF')
    parser.add_argument('--self_predictive_phi', action='store_true', help='self_predictive_phi')
    parser.add_argument('--decoupled_local_encoder', action='store_true', help='use decoupled local encoder')
    parser.add_argument('--local_for_critic', action='store_true', help='use local features for critic')
    parser.add_argument('--high_utd_for_dynamics', action='store_true', help='use high UTD for dynamics')

    parser.add_argument(
        '--actor_dims',
        nargs='+',
        type=int,
        default=[512, 512],
        help='List of hidden layer sizes for actor network'
    )
    parser.add_argument(
        '--critic_dims',
        nargs='+',
        type=int,
        default=[512, 512],
        help='List of hidden layer sizes for critic network'
    )
    parser.add_argument('--pixel_latent_dim',   type=int,   default=256,    help='Latent space dimension')

    parser.add_argument('--buffer_size',  type=int,   default=1000000,       help='Replay buffer size')
    parser.add_argument('--batch_size',   type=int,   default=256,             help='Mini-batch size')
    parser.add_argument('--relabel_ratio',  type=float,   default=0.0,               help='relabel_ratio')

    parser.add_argument('--max_ep_len',   type=int,   default=400,             help='Max steps per episode')
    parser.add_argument('--num_epochs',   type=int,   default=75_000,          help='Number of training epochs')
    parser.add_argument('--random_rollout', type=int, default=10,             help='Number of random rollouts before training')
    parser.add_argument('--train_itrs',   type=int,   default=200,          help='train iterations per epoch')
    parser.add_argument('--traj_itrs',   type=int,   default=8,          help='train per trajectory')
    parser.add_argument('--orig_utd', action='store_true', help='Use original METRA UTD training schedule')

    parser.add_argument('--env_name',     type=str,   default='ant', help='environment name')
    parser.add_argument('--frame_stack',   type=int,   default=3,          help='env frame stack')
    parser.add_argument('--obs_type',     type=str,   default='states',        help='Type of observations (e.g. states, pixels)')
    parser.add_argument('--use_rsnorm', action='store_true', help='Use RSNorm wrapper for the environment')

    # log
    parser.add_argument('--log_dir',     type=str,   default='./exp', help='log path')
    parser.add_argument('--exp_tag',     type=str,   default='', help='additional explanation tag')
    parser.add_argument('--eval_frequency',   type=int,   default=500,                    help='eval frequency')
    parser.add_argument('--num_eval_iterations',   type=int,   default=48,                help='number of evaluation episodes')
    parser.add_argument('--model_save_frequency',   type=int,   default=1000,             help='eval frequency')
    parser.add_argument('--visualize_save_frequency',   type=int,   default=500,          help='visualization frequency')
    
    # model load
    parser.add_argument('--desired_load_epoch',   type=int,   default=-1,             help='visualization frequency')
    parser.add_argument('--load_dir',     type=str,   default='./exp', help='log path')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--downstream_num_goal_steps',   type=int,   default=25,   help='Number of goal steps for downstream task')
    parser.add_argument('--goal_range',   type=float,   default=2.5,)
    parser.add_argument('--goal_epsilon',   type=float,   default=1.5)
    parser.add_argument('--goal_reset_time',   type=int,   default=200,)
    parser.add_argument('--local_phi_test',   action='store_true')
    parser.add_argument('--skill_reset_steps',   type=int,   default=999999)
    parser.add_argument('--goal_dim',   type=int,   default=2)
    parser.add_argument('--use_her',   action='store_true')
    parser.add_argument('--tight_goal_range',   action='store_true')
    
    # For maze
    parser.add_argument('--map_name',  type=str,   default='L', help='Name of the maze map')
    return parser.parse_args()



def run_train(args, epoch, env, agent, buffer, graph=None):
    obs = env.reset()
    init_obs = obs.copy() 
    # sample and normalize z for this episode
    metrics = {}
    ep_obs, ep_act, ep_rew, ep_done, ep_opt, ep_next_opt = [], [], [], [], [], []
    ep_coor, ep_goal_changed = [], []
    done = False
    while not done:
        # select action
        ep_opt.append(obs[-args.goal_dim:])
        if epoch < args.random_rollout:
            z = env.action_space.sample()
        else:
            z = agent.get_action(obs, eval=False)
        next_obs, reward, done, infos = env.step(z)
        # store for episode
        ep_obs.append(obs)
        ep_act.append(z)
        ep_rew.append(reward)
        ep_next_opt.append(obs[-args.goal_dim:])
        ep_done.append(done)
        ep_coor.append(infos['coordinates'])
        obs = next_obs
    ep_obs.append(obs)
    ep_coor.append(infos['next_coordinates'])
    
    if not infos['is_success']:
        ep_done[-1] = False

    buffer.add_episode(
        np.array(ep_obs, dtype=np.float32),
        np.array(ep_coor, dtype=np.float32),
        np.array(ep_act, dtype=np.float32),
        np.array(ep_rew, dtype=np.float32),
        np.array(ep_opt, dtype=np.float32),
        np.array(ep_next_opt, dtype=np.float32),
        np.array(ep_done, dtype=np.float32),
    )

    if epoch > args.random_rollout and epoch % args.traj_itrs == 0:
        for _ in range(args.train_itrs):
            jax_batch = buffer.sample_her(args.batch_size)
            metrics = agent.learn(jax_batch)
            
    if epoch > args.random_rollout and epoch % 10 == 0:
        print(f"[Epoch {epoch}] EnvReward: {np.sum(ep_rew):.4f}, Init Pos: {ep_coor[0]}, "
              f"Reached: {infos['coordinates']}, Goal: {obs[-args.goal_dim:]}, "
              f"Step: {len(ep_rew)}/{args.max_ep_len}")
    metrics['TrainEnvReward'] = np.sum(ep_rew)

    return metrics, np.array(ep_coor)



def run_eval(args, epoch, env, agent, binchecker) -> Dict[str, Any]:
    mean_reward = []
    eps_obs = []
    eps_coors = []
    goals = []
    eps_rew = 0
    n_epis = 48
    success_count = 0
    video_frames = []
    for e in range(n_epis):
        obs = env.reset()
        ep_obs, ep_coors, ep_rew, ep_next_obs, ep_done, ep_opt, ep_next_opt = [], [], [], [], [], [], []
        done = False
        while not done:
            if e == 0:
                video_frames.append(env.render(mode='rgb_array', height=128, width=128, camera_id=0))
            action = agent.get_action(obs, eval=True)
            next_obs, reward, done, infos = env.step(action)
            goals.append(env.cur_goal)
            ep_obs.append(obs)
            eps_rew += reward
            ep_coors.append(infos['coordinates'])
            obs = next_obs
        if infos['is_success']:
            success_count += 1

        eps_obs.append(ep_obs)
        eps_coors.append(ep_coors)
    
    print()
    print('============================== Evaluation Results ==============================')
    print(f"Epoch {epoch}, EnvReward:{eps_rew / n_epis} Reached: {infos['coordinates']}, Goal: {obs[-args.goal_dim:]}, SuccessRate: {success_count / n_epis}")
    print('================================================================================')
    print()


    eval_metrics = {
        'eval_video': wandb.Video(np.array(video_frames, dtype=np.uint8).transpose(0, 3, 1, 2), fps=20, format='mp4'),
        'EvalEnvReward': eps_rew / n_epis,
        'EvalSuccessRate': float(success_count / n_epis),
    }

    if args.goal_dim == 2:
        visualized_grid, visit_bins = binchecker.eval_visualize_grid(pad_episodes(eps_coors), goals=np.array(goals))
        eval_metrics['eval_visited_bins'] = visualized_grid
    return eval_metrics
    



if __name__ == '__main__':

    args = get_args()
    bin_bound = 70
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['XLA_FLAGS'] = "--xla_gpu_autotune_level=2"
    
    # Set JAX memory fraction based on observation type
    if args.obs_type == 'pixels':
        args.obs_dim = 64 * 64 * 3 * args.frame_stack
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.125"
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.05"
        
    np.set_printoptions(
    precision=4,      
    suppress=True,    
    floatmode='fixed')
    np.random.seed(args.seed)

    formatted_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = args.algo + '_' + args.env_name + '_' + args.exp_tag
    
    # wandb
    wandb.init(project='USRL_Downstream')
    wandb.run.name = run_name
    wandb.run.save()
    wandb.config.update(args)
    args.log_dir = os.path.join(args.log_dir, args.algo + '_' + args.exp_tag, formatted_time)
    # env
    additional_dim = 0

    from envs.custom_dmc_tasks.numeric_goal_wrapper import NumericGoalWrapper
    from envs.custom_dmc_tasks.child_policy_wrapper import ChildPolicyEnv
    from envs.custom_dmc_tasks.goal_wrappers import GoalWrapper
    from envs.custom_dmc_tasks.numeric_maze_wrapper import NumericMazeWrapper, maze_str
    eval_env = None
    if args.env_name == 'dog_goal':
        map_name = 'E'
        maze = maze_str(map_name)
        env = dmc.make('dog_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
        env = NumericMazeWrapper(
            env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        additional_dim = 2
        normalizer_mean, normalizer_std = get_normalizer_preset('dog_numeric_preset')
        normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
        normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)

    elif args.env_name == 'dog_maze':
        map_name = args.map_name  # 'L', 'H', EFS, E
        maze = maze_str(map_name)
        maze_eval = maze_str(map_name + '_eval')
        print(maze_eval)
        env = dmc.make('dog_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
        eval_env = dmc.make('dog_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze_eval, 'map_name':map_name+ '_eval'})
        env = NumericMazeWrapper(
            env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        eval_env = NumericMazeWrapper(
            eval_env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        additional_dim = 2
        normalizer_mean, normalizer_std = get_normalizer_preset('dog_numeric_preset')
        normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
        normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        eval_env = ConsistentNormalizedEnv(eval_env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    elif args.env_name == 'humanoid_run_color_goal':
        
        if args.obs_type == 'states':
            map_name = 'E'
            maze = maze_str(map_name)
            env = dmc.make('humanoid_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
            env = NumericMazeWrapper(
                env,
                goal_epsilon=args.goal_epsilon,
                max_path_length=args.max_ep_len,
                num_goal_steps=args.goal_reset_time,
            )
            additional_dim = 2
            normalizer_mean, normalizer_std = get_normalizer_preset('humanoid_numeric_preset')
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
            env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        
        elif args.obs_type == 'pixels':
            from envs.custom_dmc_tasks.pixel_wrappers_noarko import RenderWrapper, FrameStackWrapper
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
            env = GoalWrapper(
                env,
                max_path_length=args.max_ep_len,
                goal_range=args.goal_range,
                num_goal_steps=args.goal_reset_time,
                touch_end=True,
                tight_goal_range=args.tight_goal_range,
            )
            args.pixel_shape = (64, 64, 3 * args.frame_stack)
            args.pixel_dim = 64 * 64 * 3 * args.frame_stack
            env = FrameStackWrapper(env, args.frame_stack) # flattened obs shape: frame_stack * 64 * 64 * 3
            video_frames = []
            additional_dim = 2

    elif args.env_name == 'humanoid_run_maze_goal':
        from envs.custom_dmc_tasks.numeric_maze_wrapper import NumericMazeWrapper, maze_str
        map_name = args.map_name  
        maze = maze_str(map_name)
        maze_eval = maze_str(map_name + '_eval')
        print(maze_eval)
        env = dmc.make('humanoid_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
        eval_env = dmc.make('humanoid_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze_eval, 'map_name':map_name + '_eval'})

        env = NumericMazeWrapper(
            env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        eval_env = NumericMazeWrapper(
            eval_env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        additional_dim = 2
        normalizer_mean, normalizer_std = get_normalizer_preset('humanoid_numeric_preset')
        normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
        normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        eval_env = ConsistentNormalizedEnv(eval_env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)

    elif args.env_name == 'quadruped_run_color_goal':
        if args.obs_type == 'states':
            map_name = 'E'
            maze = maze_str(map_name)
            env = dmc.make('quadruped_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
            env = NumericMazeWrapper(
                env,
                goal_epsilon=args.goal_epsilon,
                max_path_length=args.max_ep_len,
                num_goal_steps=args.goal_reset_time,
            )
            additional_dim = 2
            normalizer_mean, normalizer_std = get_normalizer_preset('quadruped_numeric_preset')
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
            env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        
        elif args.obs_type == 'pixels':
            from envs.custom_dmc_tasks.pixel_wrappers_noarko import RenderWrapper, FrameStackWrapper
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
            env = GoalWrapper(
                env,
                max_path_length=args.max_ep_len,
                goal_range=args.goal_range,
                num_goal_steps=args.goal_reset_time,
                touch_end=True,
                tight_goal_range=args.tight_goal_range,
            )
            args.pixel_shape = (64, 64, 3 * args.frame_stack)
            args.pixel_dim = 64 * 64 * 3 * args.frame_stack
            env = FrameStackWrapper(env, args.frame_stack) # flattened obs shape: frame_stack * 64 * 64 * 3
            video_frames = []
            additional_dim = 2
    
    elif args.env_name == 'quadruped_run_maze_goal':
        from envs.custom_dmc_tasks.numeric_maze_wrapper import NumericMazeWrapper, maze_str
        map_name = args.map_name  
        maze = maze_str(map_name)
        maze_eval = maze_str(map_name + '_eval')
        print(maze_eval)
        env = dmc.make('quadruped_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze, 'map_name':map_name})
        eval_env = dmc.make('quadruped_run_maze', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed, task_kwargs = {'maze': maze_eval, 'map_name': map_name + '_eval'})
        env = NumericMazeWrapper(
            env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        eval_env = NumericMazeWrapper(
            eval_env,
            goal_epsilon=args.goal_epsilon,
            max_path_length=args.max_ep_len,
            num_goal_steps=args.goal_reset_time,
        )
        additional_dim = 2
        normalizer_mean, normalizer_std = get_normalizer_preset('quadruped_numeric_preset')
        normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
        normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
        eval_env = ConsistentNormalizedEnv(eval_env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    
    elif args.env_name == 'fish_goal':
        from dm_control import suite
        from envs.custom_dmc_tasks.wrappers import DMCGymWrapper
        from envs.custom_dmc_tasks.dmc import ActionRepeatWrapper
        env = suite.load(domain_name="fish", task_name="swim")
        env = ActionRepeatWrapper(env, num_repeats=2)
        env = DMCGymWrapper(env, domain="fish")
        env = NumericGoalWrapper(
            env,
            max_path_length=args.max_ep_len,
            goal_range=args.goal_range,
            num_goal_steps=args.goal_reset_time,
            touch_end=True,
            goal_dim=3,
            goal_epsilon=args.goal_epsilon,
        )
        additional_dim = 3
    
    args.goal_dim = additional_dim

    from agent.metra_sf import SF_METRA_SAC_Agent
    from agent.misl import MISL_SAC_Agent
    import pickle
    with open(os.path.join(args.load_dir, 'arg_set.pkl'), 'rb') as f:
        cp_args = pickle.load(f)

    if cp_args.algo == 'SFMETRA':
        child_agent = SF_METRA_SAC_Agent(cp_args, cp_args.obs_dim, cp_args.action_dim, cp_args.option_dim, cp_args.phi_dims, cp_args.actor_dims, cp_args.critic_dims)
    elif cp_args.algo == 'MISL':
        child_agent = MISL_SAC_Agent(cp_args, cp_args.obs_dim, cp_args.action_dim, cp_args.option_dim, cp_args.phi_dims, cp_args.actor_dims, cp_args.critic_dims)

    child_agent.load_checkpoint(args.desired_load_epoch, args.load_dir)
    
    if eval_env is None:
        eval_env = env
    eval_env = ChildPolicyEnv(
        eval_env,
        cp_dict={
            'policy': child_agent,
            'dim_option': args.option_dim,
            'discrete': False,
        },
        cp_action_range=1.5,
        cp_unit_length=True,
        cp_multi_step=args.high_level_action_freq,
        cp_num_truncate_obs=additional_dim,
        cp_omit_obs_idxs=None,
    )
    
    env = ChildPolicyEnv(
        env,
        cp_dict={
            'policy': child_agent,
            'dim_option': args.option_dim,
            'discrete': False,
        },
        cp_action_range=1.5,
        cp_unit_length=True,
        cp_multi_step=args.high_level_action_freq,
        cp_num_truncate_obs=additional_dim,
        cp_omit_obs_idxs=None,
    )


    args.obs_dim += additional_dim
    # initialize
    if args.algo == 'SAC':
        from agent.high import HierarchicalAgent
        agent = HierarchicalAgent(args, args.obs_dim, args.option_dim, args.option_dim, args.phi_dims, args.actor_dims, args.critic_dims, seed=args.seed)

    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented.")

    

    # grid coverage check
    coverage_checker = BinChecker(bound_x=bin_bound, bound_y=bin_bound, bin_size=1)

    buffer = HEREpisodeReplayBuffer(args, False, args.buffer_size, args.max_ep_len // args.high_level_action_freq, args.obs_dim, args.option_dim, args.option_dim, args.after_c_step)

    train_start_time = time.time()
    EnvInteractionSteps = 0
    train_reward = 0
    for epoch in range(args.num_epochs):
        runtime = time.time()
        metrics, ep_coordinates = run_train(args, epoch, env, agent, buffer)
        train_reward += metrics.get('TrainEnvReward', 0)
        metrics['TrainEnvReward'] = train_reward / (epoch + 1)
        EnvInteractionSteps += (args.max_ep_len)
        train_fps = (args.max_ep_len) / (time.time() - runtime)


        if epoch % args.eval_frequency == 0:
            eval_metrics = run_eval(args, epoch, eval_env, agent, coverage_checker)
            metrics.update(eval_metrics)
        coverage_matrix = coverage_checker.mark_visited(ep_coordinates)
        metrics['CoverageReward'] = coverage_matrix.sum()
        metrics['TrainFPS'] = train_fps
        metrics['EnvInteractionSteps'] = EnvInteractionSteps
        metrics['FullFPS'] = metrics['EnvInteractionSteps'] / (time.time() - train_start_time)
        
        if epoch % args.visualize_save_frequency == 0:
            wandb_image = coverage_checker.visualize_grid()
            metrics['CoverageMatrix'] = wandb_image

        for k in metrics:
            if type(metrics[k]) == jaxlib.xla_extension.ArrayImpl:
                metrics[k] = float(metrics[k])
        wandb.log(metrics)
