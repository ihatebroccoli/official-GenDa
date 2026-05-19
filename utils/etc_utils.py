from glob import glob
import re

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb


def get_latest_checkpoint(log_dir):
    """
    paths: glob으로 얻은 checkpoint 파일 경로 리스트
    return: 가장 큰 step 번호를 가진 checkpoint 경로
    """
    path = log_dir + '/*_checkpoint.pkl'
    paths = glob(path)
    # (step, path) 튜플 리스트 생성
    numbered = []
    for p in paths:
        m = re.search(r'(\d+)_checkpoint\.pkl$', p)
        if m:
            step = int(m.group(1))
            numbered.append((step, p))
    if not numbered:
        raise ValueError("No valid checkpoint files found.")
    # step 기준으로 최댓값을 가진 튜플을 골라 경로만 반환
    latest_path = max(numbered, key=lambda x: x[0])[1]
    return latest_path


def phi_plot(agent, episodes_obs):
    n_epis = len(episodes_obs)
    normed_zs = np.array([np.cos(np.linspace(0, 2 * np.pi, n_epis)), np.sin(np.linspace(0, 2 * np.pi, n_epis))]).T
    plt.figure(figsize=(5, 5))
    cmap = plt.get_cmap('viridis')
    for i in range(n_epis):
        epi_phi = agent.phi.apply(agent.phi_params, episodes_obs[i])# , agent.rng)
        plt.scatter(epi_phi[:, 0], epi_phi[:, 1], color=cmap(i/n_epis), alpha=0.5)

    # eps_obs_circle = episodes_obs.copy()
    # for circle_range in range(1, 4):
    #     eps_obs_circle[0][:n_epis][:, :2] = normed_zs * 10 * circle_range
    #     epi_phi = agent.phi.apply(agent.phi_params, eps_obs_circle[0][:n_epis]) #, agent.rng)
    #     plt.scatter(epi_phi[:, 0], epi_phi[:, 1], c=np.arange(n_epis), cmap='plasma' , alpha=0.5, marker='v')
    plt.axis('equal')
    image = wandb.Image(plt.gcf())
    plt.close()

    return image