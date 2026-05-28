import copy
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Sequence, Tuple
from functools import partial
import numpy as np

from agent.networks import get_ensemble, GaussianActor, Critic

@jax.jit
def get_alpha(log_alpha):
    return jnp.exp(log_alpha)
    
class HighLevelSAC:
    def __init__(self, args, obs_dim: int, action_dim: int, option_dim:int,
                 phi_dim: int, actor_dims: Sequence[int],
                 critic_dims: Sequence[int], seed: int = 0,
                 lr: float = 1e-4, tau: float = 0.005, gamma: float = 0.99, n_critics: int = 2):
        self.gamma = gamma
        self.tau = tau
        self.rng = jax.random.PRNGKey(seed)
        self.target_entropy = -(option_dim)
        self.args = args
        pixel_shape = args.pixel_shape
        self.pixel_dim = args.pixel_dim
        # Core networks
        self.actor = GaussianActor([512, 512], option_dim,
                                   use_encoder=(args.obs_type in ('pixels', 'hybrid')), 
                                   pix_latent_dim=args.pixel_latent_dim,
                                   pixel_dim=self.pixel_dim,
                                   pixel_shape=pixel_shape,)
        self.critic = get_ensemble(n_critics, Critic, methods=['__call__'])([512, 512], 1, 
                                   use_encoder=(args.obs_type in ('pixels', 'hybrid')),
                                   pix_latent_dim=args.pixel_latent_dim,
                                   pixel_dim=args.pixel_dim,
                                   pixel_shape=pixel_shape,)

        # initialize parameters
        dummy_s  = jnp.zeros((1, obs_dim))
        dummy_a  = jnp.zeros((1, option_dim))
        
        self.rng, *keys = jax.random.split(self.rng, 4)
        k_actor, k_c1, k_c2, = keys

        self.actor_params  = self.actor.init(k_actor, dummy_s, k_actor)
        self.critic_params = self.critic.init(k_c1, dummy_s, dummy_a)
        self.critic_target_params = self.critic_params.copy()
        self.log_alpha = jnp.log(0.01)
        
        # optimizers
        self.opt_actor   = optax.adam(1e-4)
        self.opt_critic  = optax.adam(3e-4)
        self.opt_alpha   = optax.adam(1e-3)
        
        self.opt_actor_state  = self.opt_actor.init(self.actor_params)
        self.opt_critic_state = self.opt_critic.init(self.critic_params)
        self.opt_alpha_state   = self.opt_alpha.init(self.log_alpha)
        

    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, actor_param, s, key):
        a, _ = self.actor.apply(actor_param, s, key)
        return a[0]


    @partial(jax.jit, static_argnums=(0,))
    def _get_eval_action(self, actor_param, s):
        a = self.actor.apply(actor_param, s, method='eval_action')
        return a[0]
        

    def get_action(self, s, eval=False):
        s = jnp.asarray(s[None]).astype(jnp.float32)
        
        if eval:
            return np.array(self._get_eval_action(self.actor_params, s))
        self.rng, key = jax.random.split(self.rng)
        return np.array(self._get_action(self.actor_params, s, key))


    def learn(self, batch, batch_size=256):
        self.rng, key = jax.random.split(self.rng)
        params, states, metrics = self.update(
         self.actor_params, self.critic_params, self.critic_target_params, self.log_alpha, 
         self.opt_actor_state, self.opt_critic_state, self.opt_alpha_state, 
         batch, key)
        (self.actor_params, self.critic_params, self.critic_target_params,
         self.log_alpha) = params
        (self.opt_actor_state, self.opt_critic_state, self.opt_alpha_state) = states
        return metrics
    
    
    @partial(jax.jit, static_argnums=(0,))
    def update(self, actor_params, critic_params, critic_target_params, log_alpha, 
               opt_actor_state, opt_critic_state, opt_alpha_state, 
               batch, key):
        sg, a, r, sg2, d = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']
        key, key_targ, key_pi = jax.random.split(key, 3)
        # critic
        a2, logp2 = self.actor.apply(actor_params, sg2, key_targ)
        alpha = get_alpha(log_alpha)
        backup = r + (1. - d) * (self.gamma * (self.critic.apply(critic_target_params, sg2, a2).min(axis=0) - alpha * logp2.squeeze()))
        def critic_loss_fn(c_p):
            q = self.critic.apply(c_p, sg, a)
            loss = 0.5 * ((q - backup)**2).mean()
            return loss, loss
        

        def actor_loss_fn(actor_p):
            a_pred, logp = self.actor.apply(actor_p, sg, key_pi)
            q = self.critic.apply(critic_params, sg, a_pred).min(axis=0)
            loss = (alpha * logp.squeeze() - q).mean() 
            return loss, (alpha, q.min(), logp)


        def alpha_loss_fn(log_a, logp):
            loss = -log_a * (logp + self.target_entropy).mean()
            return loss, loss

        grads_c, q_loss = jax.grad(critic_loss_fn, has_aux=True)(critic_params)
        updates_c, opt_critic_state = self.opt_critic.update(grads_c, opt_critic_state)
        critic_params = optax.apply_updates(critic_params, updates_c)

        (actor_loss, (alpha, actor_q_min, logp)), grads_actor =\
             jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
        grads_alpha, alpha_loss = jax.grad(alpha_loss_fn, argnums=(0), has_aux=True)(log_alpha, logp)

        updates_a, opt_actor_state = self.opt_actor.update(grads_actor, opt_actor_state)
        actor_params = optax.apply_updates(actor_params, updates_a)

        updates_alpha, opt_alpha_state = self.opt_alpha.update(grads_alpha, opt_alpha_state)
        log_alpha = optax.apply_updates(log_alpha, updates_alpha)
        critic_target_params = optax.incremental_update(critic_params, critic_target_params, self.tau)
        metrics = {
            'high_critic_loss': q_loss,
            'high_actor_loss': actor_loss, 
            'high_alpha': alpha, 
            'high_actor_q_min': actor_q_min, 
            'high_actor_logp': logp.mean(),
            'high_alpha_loss': alpha_loss, 
            'rew_mean': r.mean(),
        }

        new_states = (opt_actor_state, opt_critic_state, opt_alpha_state)
        new_params = (actor_params, critic_params, critic_target_params, log_alpha)
        return new_params, new_states, metrics



class HierarchicalAgent:
    def __init__(self, args, obs_dim: int, action_dim: int, option_dim:int,
                 phi_dim: int, actor_dims: Sequence[int],
                 critic_dims: Sequence[int], seed: int = 0,
                 lr: float = 1e-4, tau: float = 0.005, gamma: float = 0.99):
        self.high_agent = HighLevelSAC(args, obs_dim, action_dim, option_dim,
                                       phi_dim, actor_dims, critic_dims,
                                       seed, lr, tau, gamma)


    def get_action(self, s, eval=False):
        ha = self.high_agent.get_action(s, eval)
        return ha


    def learn(self, batch):
        high_metrics = self.high_agent.learn(batch)
        return high_metrics
