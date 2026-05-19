from typing import Dict, Tuple, Any
import numpy as np
import jax.numpy as jnp
from utils.etc_utils import phi_plot
import wandb
import time
import matplotlib.pyplot as plt

def run_train(args, epoch, env, agent, buffer, graph=None) -> Tuple[Dict[str, Any], np.ndarray]:
    metrics = {}
    obs = env.reset()
        
    random_z = np.random.normal(size=args.option_dim).astype(np.float32)
    normed_z = agent._skill_preprocess(random_z)

    ep_obs, ep_act, ep_rew, ep_done, ep_opt, ep_next_opt = [], [], [], [], [], []
    ep_coor = []
    
    for t in range(args.max_ep_len):
        # select action
        if epoch < args.random_rollout:
            action = env.action_space.sample()
        else:
            action = agent.get_action(obs, normed_z)
        next_obs, reward, done, infos = env.step(action)

        # store for episode
        ep_obs.append(obs)
        ep_act.append(action)
        ep_rew.append(reward)
        ep_opt.append(normed_z)
        ep_next_opt.append(normed_z)
        ep_done.append(done)
        ep_coor.append(infos['coordinates'])
        obs = next_obs
        if done:
            break
    ep_obs.append(obs)

    buffer.add_episode(
        np.array(ep_obs, dtype=np.float32),
        np.array(ep_act, dtype=np.float32),
        np.array(ep_rew, dtype=np.float32),
        np.array(ep_opt, dtype=np.float32),
        np.array(ep_next_opt, dtype=np.float32),
        np.array(ep_done, dtype=np.float32),
    )

    # train
    if epoch > args.random_rollout and epoch % args.traj_itrs == 0:
        for _ in range(args.train_itrs):
            jax_batch = buffer.sample(args.batch_size)
            agent_metrics = agent.learn(jax_batch, epoch)
            metrics.update(agent_metrics)
            
    if epoch > args.random_rollout and epoch % 10 == 0:
        print(f"Epoch {epoch}, EnvReward:{np.sum(ep_rew)} Reached: {infos['coordinates']},  Metrics: {metrics}")


    return metrics, np.array(ep_coor)


def run_eval(args, epoch, env, agent, binchecker) -> Dict[str, Any]:
    mean_reward = []
    eps_obs = []
    eps_coors = []
    n_epis = args.num_eval_iterations
    video_frames = []
    reward_sum = 0.
    for e in range(n_epis):
        obs = env.reset()
        random_z = np.random.normal(size=args.option_dim).astype(np.float32)
        normed_z = agent._skill_preprocess(random_z)
        
        ep_obs, ep_coors, ep_rew, ep_next_obs, ep_done, ep_opt, ep_next_opt = [], [], [], [], [], [], []
        for t in range(args.max_ep_len):
            if e == 0:
                video_frames.append(env.render(mode='rgb_array', height=92, width=92, camera_id=0))
            action = agent.get_action(obs, normed_z, eval=True)
            next_obs, reward, done, infos = env.step(action)
            reward_sum += reward
            obs = next_obs
            
            if done:
                ep_obs.append(np.zeros_like(obs))
                ep_coors.append(np.zeros_like(infos['coordinates']))
                continue
            else:
                ep_obs.append(obs)
                ep_coors.append(infos['coordinates'])

        eps_obs.append(ep_obs)
        eps_coors.append(ep_coors)
    visualized_grid, visit_bins = binchecker.eval_visualize_grid(np.array(eps_coors))
    print()
    print('============================== Evaluation Results ==============================')
    print(f"Epoch {epoch}, Reward: {visit_bins}")
    print('================================================================================')
    print()

    def cast_obs(x):
        x = np.array(x)
        x = jnp.asarray(x)  
        if args.obs_type == 'pixels':
            x = x.astype(jnp.float32)
        return x

    eval_metrics = {
        'eval_env_avg_reward': reward_sum / n_epis,
        'eval_mean_reward': visit_bins,
        'eval_visited_bins': visualized_grid,
        'eval_phi_plot': phi_plot(agent, cast_obs(eps_obs)),
        'eval_video': wandb.Video(np.array(video_frames, dtype=np.uint8).transpose(0, 3, 1, 2), fps=20, format='mp4'),
    }
    return eval_metrics
    
