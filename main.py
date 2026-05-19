from envs.custom_dmc_tasks import dmc
from utils.replay_buffer import EpisodeReplayBuffer
from utils.env_utils import BinChecker, ConsistentNormalizedEnv, get_normalizer_preset

from run import run_train, run_eval
import jaxlib, os, jax
from jax import lax

import argparse
import datetime, time

import wandb
import numpy as np
from pathlib import Path
import pickle

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
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    parser.add_argument('--local_latent_dim',     type=int,   default=27,    help='Phi network output dimension')
    parser.add_argument('--z_relabeling', action='store_true', help='relabeling options with final - initi observations')
    parser.add_argument('--csf_weight_low_bound',     type=float,   default=0.1,    help='Phi network output dimension')
    parser.add_argument('--after_c_step',     type=int,   default=10,    help='Phi network output dimension')
    
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
    
    parser.add_argument('--local_phi_test', action='store_true', help='local_phi_test')

    parser.add_argument(
        '--actor_dims',
        nargs='+',
        type=int,
        default=[1024, 1024],
        help='List of hidden layer sizes for actor network'
    )
    parser.add_argument(
        '--critic_dims',
        nargs='+',
        type=int,
        default=[1024, 1024],
        help='List of hidden layer sizes for critic network'
    )
    parser.add_argument('--pixel_latent_dim',   type=int,   default=256,    help='Latent space dimension')

    # Buffer settings
    parser.add_argument('--buffer_size',  type=int,   default=1000000,       help='Replay buffer size')
    parser.add_argument('--batch_size',   type=int,   default=256,             help='Mini-batch size')

    # learning settings
    parser.add_argument('--max_ep_len',   type=int,   default=400,             help='Max steps per episode')
    parser.add_argument('--num_epochs',   type=int,   default=75_000,          help='Number of training epochs')
    parser.add_argument('--random_rollout', type=int, default=10,             help='Number of random rollouts before training')
    parser.add_argument('--train_itrs',   type=int,   default=200,          help='train iterations per epoch')
    parser.add_argument('--traj_itrs',   type=int,   default=8,          help='train per trajectory')
    parser.add_argument('--orig_utd', action='store_true', help='Use original METRA UTD training schedule')

    parser.add_argument('--env_name',     type=str,   default='ant', help='environment name')
    parser.add_argument('--frame_stack',   type=int,   default=3,          help='env frame stack')
    parser.add_argument('--obs_type',     type=str,   default='states',        help='Type of observations (e.g. states, pixels)')

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


    parser.add_argument('--skill_reset_steps',   type=int,   default=1000000,          help='number of skill changing steps')
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--use_mmd', action='store_true')
    parser.add_argument('--use_dir_penalty', action='store_true')

    parser.add_argument('--reset_ac_cr', action='store_true')
    parser.add_argument('--reset_epoch_freq',   type=int,   default=1250,          help='goal reset time for ant multigoal')
    
    parser.add_argument('--discrete', action='store_true')

    return parser.parse_args()


import random
if __name__ == '__main__':
    args = get_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    bin_bound = 70
    bin_size = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['XLA_FLAGS'] = "--xla_gpu_autotune_level=2"
    
    # Set JAX memory fraction based on observation type
    if args.obs_type == 'pixels' or args.obs_type == 'hybrid':
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.20"
        if args.obs_type == 'pixels':
            args.obs_dim = args.frame_stack * args.obs_dim * args.obs_dim * 3
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.10"

    np.set_printoptions(
        precision=4,    
        suppress=True,  
        floatmode='fixed')

    formatted_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = args.algo + '_' + args.env_name + '_' + args.exp_tag
    
    # wandb
    wandb.init(project='USRL_v2')
    wandb.run.name = run_name
    wandb.run.save()
    wandb.config.update(args)
    args.log_dir = os.path.join(args.log_dir, args.algo + '_' + args.exp_tag, formatted_time)

    # env
    env_normalizer = None
    if args.env_name == 'fish':
        from dm_control import suite
        from envs.custom_dmc_tasks.wrappers import DMCGymWrapper
        from envs.custom_dmc_tasks.dmc import ActionRepeatWrapper
        bin_bound = 0.3
        bin_size = 0.01
        env = suite.load(domain_name="fish", task_name="swim")
        env = ActionRepeatWrapper(env, num_repeats=2)
        env = DMCGymWrapper(env, domain="fish")
        normalizer_mean, normalizer_std = get_normalizer_preset('fish_preset')
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    
    elif args.env_name == 'humanoid_run_color':
        if args.obs_type == 'pixels':
            from envs.custom_dmc_tasks.pixel_wrappers_noarko import RenderWrapper, FrameStackWrapper
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
            env = FrameStackWrapper(env, args.frame_stack) # flattened obs shape: frame_stack * 64 * 64 * 3
            args.pixel_shape = (64, 64, 3 * args.frame_stack)
            args.pixel_dim = 64 * 64 * 3 * args.frame_stack
        else:
            env = dmc.make(args.env_name, obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            normalizer_mean, normalizer_std = get_normalizer_preset('humanoid_numeric_preset')
            env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    
    elif args.env_name == 'dog_run':
        from dm_control import suite
        from envs.custom_dmc_tasks.wrappers import DMCGymWrapper
        from envs.custom_dmc_tasks.dmc import ActionRepeatWrapper
        env = suite.load(domain_name="dog", task_name="run")
        env = ActionRepeatWrapper(env, num_repeats=2)
        env = DMCGymWrapper(env, domain="dog")
        normalizer_mean, normalizer_std = get_normalizer_preset('dog_numeric_preset')
        env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    
    elif args.env_name == 'kitchen':
        import sys
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)

    elif args.env_name == 'quadruped_run_forward_color':
        if args.obs_type == 'pixels':
            from envs.custom_dmc_tasks.pixel_wrappers_noarko import RenderWrapper, FrameStackWrapper
            env = dmc.make(args.env_name, obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
            env = FrameStackWrapper(env, args.frame_stack) # flattened obs shape: frame_stack * 64 * 64 * 3
            args.pixel_shape = (64, 64, 3 * args.frame_stack)
            args.pixel_dim = 64 * 64 * 3 * args.frame_stack
        else:
            env = dmc.make(args.env_name, obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            normalizer_mean, normalizer_std = get_normalizer_preset('quadruped_numeric_preset')
            env = ConsistentNormalizedEnv(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std)
    else:
        raise NotImplementedError(f"Environment {args.env_name} is not implemented.")

    # grid coverage check
    coverage_checker = BinChecker(bound_x=bin_bound, bound_y=bin_bound, bin_size=bin_size)

    # initialize
    if args.algo == 'SFMETRA':
        from agent.genda import GenDaAgent
        agent = GenDaAgent(args, args.obs_dim, args.action_dim, args.option_dim, args.phi_dims, args.actor_dims, args.critic_dims, seed=args.seed, env_normalizer=env_normalizer)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented.")
    
    buffer = EpisodeReplayBuffer(args, args.obs_type == 'pixels', args.buffer_size, args.max_ep_len, args.obs_dim, args.action_dim, args.option_dim, args.after_c_step, use_mmd=args.use_mmd)

    if args.load_model:
        desired_epoch = None
        agent.load_checkpoint(args.desired_load_epoch, args.load_dir)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(f'{args.log_dir}/arg_set.pkl', 'wb') as f:
        checkpoint = pickle.dump(args, f)

    train_start_time = time.time()
    EnvInteractionSteps = 0

    for epoch in range(args.num_epochs):
        runtime = time.time()
        if epoch > args.random_rollout:
            metrics, ep_coordinates = run_train(args, epoch, env, agent, buffer)
        else:
            metrics, ep_coordinates = run_train(args, epoch, env, agent, buffer)
        
        if epoch % args.reset_epoch_freq == 0 and epoch > 0 and args.reset_ac_cr:
            print("Resetting actor and critic networks.")
            agent.shrkpertb_actor_critic()

        EnvInteractionSteps += args.max_ep_len
        train_fps = args.max_ep_len / (time.time() - runtime)


        if epoch % args.eval_frequency == 0:
            eval_metrics = run_eval(args, epoch, env, agent, coverage_checker)
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

        if epoch % args.model_save_frequency == 0:
            agent.save_checkpoint(epoch, args.log_dir)
