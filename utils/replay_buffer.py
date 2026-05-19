import numpy as np
import jax.numpy as jnp
import time



class HEREpisodeReplayBuffer:
    def __init__(self, args, pixel_env, buffer_size, max_episode_len, obs_dim, action_dim, option_dim, after_c_step, use_mmd=False):
        self.buffer_size = buffer_size
        self.args = args
        self.max_episode_len = max_episode_len
        self.capacity = buffer_size // max_episode_len

        self.obs_type = np.float32
        self.obs_buf = np.zeros((self.capacity, max_episode_len + 1, obs_dim), dtype=self.obs_type)
        self.coor_buf = np.zeros((self.capacity, max_episode_len + 1, 2), dtype=np.float32)

        self.option_buf = np.zeros((self.capacity, max_episode_len, option_dim), dtype=np.float32)
        self.next_option_buf = np.zeros((self.capacity, max_episode_len, option_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity, max_episode_len + 1, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.capacity, max_episode_len), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, max_episode_len), dtype=np.float32)
        self.use_mmd = use_mmd

        self.len_buf = np.zeros((self.capacity,), dtype=np.int32)

        self.ptr = 0
        self.size = 0
        self.after_c_step = after_c_step


    def add_episode(self, obs, coor, act, rew, options, next_options, done):
        T = min(len(rew), self.max_episode_len)
        assert obs.shape[0] >= T + 1, "obs must include T+1 frames"

        o = np.zeros((self.max_episode_len + 1, obs.shape[-1]), dtype=self.obs_type)
        o[:T+1] = obs[:T+1]
        coor_ = np.zeros((self.max_episode_len + 1, 2), dtype=np.float32)
        coor_[:T+1] = coor[:T+1]

        a = np.zeros((self.max_episode_len + 1, act.shape[-1]), dtype=np.float32); a[:T] = act[:T]
        ops = np.zeros((self.max_episode_len, options.shape[-1]), dtype=np.float32); ops[:T] = options[:T]
        nops = np.zeros_like(ops); nops[:T] = next_options[:T]
        r = np.zeros((self.max_episode_len,), dtype=np.float32); r[:T] = rew[:T]
        d = np.zeros((self.max_episode_len,), dtype=np.float32); d[:T] = done[:T]
        idx = self.ptr
        self.obs_buf[idx] = o
        self.coor_buf[idx] = coor_
        self.act_buf[idx] = a
        self.rew_buf[idx] = r
        self.done_buf[idx] = d
        self.option_buf[idx] = ops
        self.next_option_buf[idx] = nops
        self.len_buf[idx] = T

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def cast_obs(self, x):
        x = jnp.asarray(x)  
        if self.obs_type == np.uint8:
            x = x.astype(jnp.float32)
        return x

    def sample_her(self, batch_size:int,):
        assert self.size > 0
        epi_idxs = np.random.choice(self.size, batch_size, replace=True)

        len_per_epi = self.len_buf[epi_idxs]  # T
        step_idxs = np.array([np.random.randint(0, L) for L in len_per_epi])

        allowed = np.maximum(len_per_epi - step_idxs - 1, 0)       
    
        future_t = np.array([
            np.random.randint(0, m+1)
            for m in allowed
        ])

        init_idx = np.zeros_like(future_t)
        last_idx = np.array(len_per_epi)
        
        obs = np.array(self.obs_buf[epi_idxs, step_idxs], copy=True)
        coors = np.array(self.coor_buf[epi_idxs, step_idxs], copy=True)
        next_obs = np.array(self.obs_buf[epi_idxs, step_idxs + 1], copy=True)
        next_coors = np.array(self.coor_buf[epi_idxs, step_idxs + 1], copy=True)
        rew = np.array(self.rew_buf[epi_idxs, step_idxs], copy=True)
        done = np.array(self.done_buf[epi_idxs, step_idxs], copy=True)

        if self.args.use_her:
            masks = np.random.uniform(0, 1, size=(batch_size,)) < 0.8
            new_goals = self.coor_buf[epi_idxs, step_idxs + future_t + 1]
            new_rews = np.linalg.norm(next_coors[:] - new_goals, axis=-1) <= self.args.goal_epsilon
            obs[:, -self.args.goal_dim:] = np.where(masks[:, None], new_goals, obs[:, -self.args.goal_dim:])
            next_obs[:, -self.args.goal_dim:] = np.where(masks[:, None], new_goals, next_obs[:, -self.args.goal_dim:])

            rew = np.where(masks, new_rews, rew)
            done = np.where(masks, new_rews, done)

        mini_batch = {
            'obs':         self.cast_obs(obs),
            'next_obs':    self.cast_obs(next_obs),
            'act':         jnp.asarray(self.act_buf[epi_idxs, step_idxs]),
            'next_act':    jnp.asarray(self.act_buf[epi_idxs, step_idxs + 1]), 
            'rew':         jnp.asarray(rew),
            'done':        jnp.asarray(done),
            'options':     jnp.asarray(self.option_buf[epi_idxs, step_idxs]),
            'next_options':jnp.asarray(self.next_option_buf[epi_idxs, step_idxs]),
            'ep_init_obs': self.cast_obs(self.obs_buf[epi_idxs, init_idx]),
            'ep_last_obs': self.cast_obs(self.obs_buf[epi_idxs, last_idx]),  
        }
        
        return mini_batch



class EpisodeReplayBuffer:
    def __init__(self, args, pixel_env, buffer_size, max_episode_len, obs_dim, action_dim, option_dim, after_c_step, use_mmd=False):
        self.buffer_size = buffer_size
        self.args = args
        self.max_episode_len = max_episode_len
        self.capacity = buffer_size // max_episode_len

        self.obs_type = np.float32 if not pixel_env else np.uint8
        self.obs_buf = np.zeros((self.capacity, max_episode_len + 1, obs_dim), dtype=self.obs_type)

        self.option_buf = np.zeros((self.capacity, max_episode_len, option_dim), dtype=np.float32)
        self.next_option_buf = np.zeros((self.capacity, max_episode_len, option_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity, max_episode_len + 1, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.capacity, max_episode_len), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, max_episode_len), dtype=np.float32)
        self.use_mmd = use_mmd

        self.len_buf = np.zeros((self.capacity,), dtype=np.int32)

        self.ptr = 0
        self.size = 0
        self.after_c_step = after_c_step


    def add_episode(self, obs, act, rew, options, next_options, done):
        T = min(len(rew), self.max_episode_len)
        assert obs.shape[0] >= T + 1, "obs must include T+1 frames"

        o = np.zeros((self.max_episode_len + 1, obs.shape[-1]), dtype=self.obs_type)
        o[:T+1] = obs[:T+1]

        a = np.zeros((self.max_episode_len + 1, act.shape[-1]), dtype=np.float32); a[:T] = act[:T]
        ops = np.zeros((self.max_episode_len, options.shape[-1]), dtype=np.float32); ops[:T] = options[:T]
        nops = np.zeros_like(ops); nops[:T] = next_options[:T]
        r = np.zeros((self.max_episode_len,), dtype=np.float32); r[:T] = rew[:T]
        d = np.zeros((self.max_episode_len,), dtype=np.float32); d[:T] = done[:T]
        idx = self.ptr
        self.obs_buf[idx] = o
        self.act_buf[idx] = a
        self.rew_buf[idx] = r
        self.done_buf[idx] = d
        self.option_buf[idx] = ops
        self.next_option_buf[idx] = nops
        self.len_buf[idx] = T

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def cast_obs(self, x):
        x = jnp.asarray(x)  
        if self.obs_type == np.uint8:
            x = x.astype(jnp.float32)
        return x

    def sample(self, batch_size:int, p:float=0.999, k_min=3, k_max=5):
        assert self.size > 0
        epi_idxs = np.random.choice(self.size, batch_size, replace=True)

        len_per_epi = self.len_buf[epi_idxs]  # T
        step_idxs = np.array([np.random.randint(0, L) for L in len_per_epi])

        allowed = np.maximum(len_per_epi - step_idxs - 1, 0)      
        max_offsets = np.minimum(k_max, allowed)       
        uniform_offsets = np.array([
            np.random.randint(1, m + 1) if m > 0 else 1
            for m in max_offsets
        ])
        future_t = step_idxs + uniform_offsets

        init_idx = np.zeros_like(future_t)
        last_idx = np.array(len_per_epi)

        mini_batch = {
            'obs':         self.cast_obs(self.obs_buf[epi_idxs, step_idxs]),
            'next_obs':    self.cast_obs(self.obs_buf[epi_idxs, step_idxs + 1]),
            'act':         jnp.asarray(self.act_buf[epi_idxs, step_idxs]),
            'next_act':    jnp.asarray(self.act_buf[epi_idxs, step_idxs + 1]),
            'rew':         jnp.asarray(self.rew_buf[epi_idxs, step_idxs]),
            'done':        jnp.asarray(self.done_buf[epi_idxs, step_idxs]),
            'options':     jnp.asarray(self.option_buf[epi_idxs, step_idxs]),
            'next_options':jnp.asarray(self.next_option_buf[epi_idxs, step_idxs]),
            'ep_init_obs': self.cast_obs(self.obs_buf[epi_idxs, init_idx]),
            'ep_last_obs': self.cast_obs(self.obs_buf[epi_idxs, last_idx]),  
            'after_c_obs': self.cast_obs(self.obs_buf[epi_idxs, future_t]),
            'c_act':       jnp.asarray(self.act_buf[epi_idxs, future_t]),
        }

        return mini_batch
