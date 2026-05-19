import copy
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Sequence, Tuple
from functools import partial
import numpy as np
import os
import time
from pathlib import Path

from agent.networks import get_ensemble, GaussianActor, Critic, PseudoLocalEncoder, \
    DynamicsModel, DistributionalModel, Encoder, Decoder, MLPcount, \
    init_lsh_preprocess, preprocess_lsh
from utils.etc_utils import get_latest_checkpoint


@jax.jit
def get_kl_loss(mu, logstd):
    logvar = 2.0 * logstd
    var    = jnp.exp(logvar)
    kl_per_ex = 0.5 * jnp.sum(var + mu**2 - 1.0 - logvar + 1e-8, axis=-1)
    kl_loss = kl_per_ex.mean()
    return kl_loss

@jax.jit
def get_alpha(log_alpha):
    return jnp.exp(log_alpha)

@jax.jit
def get_lambda(log_lambda):
    return jnp.exp(log_lambda)

class GenDaAgent:
    def __init__(self, args, obs_dim: int, action_dim: int, option_dim:int, phi_dims: Sequence[int],
                 actor_dims: Sequence[int], critic_dims: Sequence[int], seed: int = 0,
                 lr: float = 1e-4, lag_lr: float = 1e-4, tau: float = 0.005, gamma: float = 0.99,
                 target_entropy: float = None, n_critics: int = 2, env_normalizer=None):
        self.args = args
        self.gamma = gamma
        self.tau = tau
        self.rng = jax.random.PRNGKey(seed)
        self.target_entropy = -np.prod(action_dim).item()
        self.option_dim = option_dim
        
        if args.obs_type == 'states':
            self.local_kl_coeff = 0.1
            pixel_shape = None
            self.pixel_dim=None
            self.local_latent_dim = args.local_latent_dim
        else:
            self.local_kl_coeff = 0.1
            pixel_shape = args.pixel_shape
            self.pixel_dim = args.pixel_dim
            self.local_latent_dim = args.local_latent_dim

        self.phi = Encoder(phi_dims, use_encoder=(args.obs_type == 'pixels' or args.obs_type == 'hybrid'), 
                                    pix_latent_dim=args.pixel_latent_dim,
                                    pixel_shape=pixel_shape, pixel_dim=self.pixel_dim)

        self.actor = GaussianActor(actor_dims, action_dim, 
                                   use_encoder=(args.obs_type == 'pixels' or args.obs_type == 'hybrid'), 
                                   pix_latent_dim=args.pixel_latent_dim,
                                   pixel_dim=self.pixel_dim,
                                   pixel_shape=pixel_shape,
                                   use_local=args.with_local_state)
        self.critic = get_ensemble(n_critics, Critic, methods=['__call__'])(critic_dims, 1, 
                                   use_encoder=(args.obs_type == 'pixels' or args.obs_type == 'hybrid'), 
                                   use_local=args.local_for_critic,
                                   pix_latent_dim=args.pixel_latent_dim,
                                   pixel_dim=self.pixel_dim,
                                   pixel_shape=pixel_shape,
                                   )

        dummy_s = jnp.zeros((1, obs_dim + option_dim))
        dummy_z  = jnp.zeros((1, option_dim))
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, action_dim))

        self.rng, *keys = jax.random.split(self.rng, 8)
        k_phi, k_actor, k_c, k_alpha, k_lambda, k_prior, k_decoder = keys
        self.phi_params = self.phi.init(k_phi, dummy_obs)
        self.target_phi_params = copy.deepcopy(self.phi_params)
        if args.with_local_state:
            dummy_ls = jnp.zeros((1, self.local_latent_dim + option_dim))
            self.actor_params = self.actor.init(k_actor, dummy_ls, k_actor)
        else:
            self.actor_params = self.actor.init(k_actor, dummy_s, k_actor)

        if args.local_for_critic:
            self.critic_params = self.critic.init(k_c, dummy_ls, dummy_act)
        else:
            self.critic_params = self.critic.init(k_c, dummy_s, dummy_act)
        self.critic_target_params = self.critic_params.copy()

        self.log_alpha = jnp.log(0.01)
        self.log_lambda = jnp.log(30.)
        self.log_neg_lambda = jnp.log(1.)

        self.opt_phi = optax.adam(lr)
        self.opt_actor = optax.adam(lr)
        self.opt_critic = optax.adam(lr)
        self.opt_alpha = optax.adam(lag_lr)
        self.opt_lambda = optax.adam(lag_lr)

        self.opt_phi_state = self.opt_phi.init(self.phi_params)
        self.opt_target_phi_state = self.opt_phi.init(self.target_phi_params)
        self.opt_lambda_state = self.opt_lambda.init(self.log_lambda)
        self.opt_neg_lambda_state = self.opt_lambda.init(self.log_neg_lambda)

        self.opt_actor_state = self.opt_actor.init(self.actor_params)
        self.opt_critic_state = self.opt_critic.init(self.critic_params)
        self.opt_alpha_state = self.opt_alpha.init(self.log_alpha)
        self.dual_slack = 1e-3

        # Local Encoder
        self.local_encoder_tau = 0.01
        
        dummy_local_latent = jnp.zeros((1, self.local_latent_dim))
        dummy_decoder_input = jnp.zeros((1, self.local_latent_dim + self.option_dim))
        dummy_option = jnp.zeros((1, option_dim ))
        self.rng, k_le, k_ld, k_ge, k_gd, k_dm = jax.random.split(self.rng, 6)

        self.local_encoder = DistributionalModel(self.local_latent_dim, use_encoder=args.obs_type == 'pixels' or args.obs_type == 'hybrid', pixel_dim=self.pixel_dim,
                                                                pix_latent_dim=self.local_latent_dim, pixel_shape=args.pixel_shape) if not args.pseudo_local_encoder else PseudoLocalEncoder([0, 1])
        self.local_decoder = Decoder(obs_dim, use_encoder=args.obs_type == 'pixels' or args.obs_type == 'hybrid',
                                     pix_latent_dim=args.pixel_latent_dim, pixel_shape=args.pixel_shape)
        self.local_log_decoder = Decoder(obs_dim, use_encoder=args.obs_type == 'pixels' or args.obs_type == 'hybrid',
                                     pix_latent_dim=args.pixel_latent_dim, pixel_shape=args.pixel_shape)
        self.global_decoder = Decoder(obs_dim, use_encoder=args.obs_type == 'pixels' or args.obs_type == 'hybrid',
                                     pix_latent_dim=args.pixel_latent_dim, pixel_shape=args.pixel_shape)
        
        self.dynamics_model = DynamicsModel(self.local_latent_dim)
        
        self.local_encoder_params = self.local_encoder.init(k_le, dummy_obs)
        self.local_encoder_loglambda = jnp.log(0.0001)
        self.target_local_encoder_params = copy.deepcopy(self.local_encoder_params)
        self.local_decoder_params = self.local_decoder.init(k_ld, dummy_decoder_input)
        self.local_decoder_log_params = self.local_log_decoder.init(k_ld, dummy_local_latent)
        self.global_decoder_params = self.global_decoder.init(k_ge, dummy_option)
        self.dynamics_model_params = self.dynamics_model.init(k_dm, dummy_local_latent, dummy_act)

        self.opt_local_encoder = optax.adam(lr)
        self.opt_local_decoder = optax.adam(lr)
        self.opt_global_decoder = optax.adam(lr)
        self.opt_dynamics_model = optax.adam(lr)
        self.opt_local_encoder_state = self.opt_local_encoder.init(self.local_encoder_params)
        self.opt_local_loglambda_state = self.opt_lambda.init(self.local_encoder_loglambda)
        self.opt_local_decoder_state = self.opt_local_decoder.init(self.local_decoder_params)
        self.opt_local_decoder_log_state = self.opt_local_decoder.init(self.local_decoder_log_params)
        self.opt_global_decoder_state = self.opt_local_decoder.init(self.global_decoder_params)
        self.opt_dynamics_model_state = self.opt_dynamics_model.init(self.dynamics_model_params)

        # phi_contrastive target test
        self.local_phi = Critic(critic_dims, option_dim)
        self.local_phi_params = self.local_phi.init(k_dm, dummy_local_latent, dummy_act)
        self.target_local_phi_params = copy.deepcopy(self.local_phi_params)
        self.opt_local_phi = optax.adam(lr)
        self.opt_local_phi_state = self.opt_local_phi.init(self.local_phi_params)

        # count for z
        self.z_count = MLPcount()
        self.z_count_params = self.z_count.init(k_gd, jnp.zeros((1, 256)))
        self.opt_z_count = optax.adam(0.01)
        self.opt_z_count_state = self.opt_z_count.init(self.z_count_params)
        self.lsh_H = init_lsh_preprocess(self.option_dim, k_prior)


    def obs_preprocess(self, s: jnp.ndarray) -> jnp.ndarray:
        if self.args.obs_type == 'hybrid':
            return s.at[:, self.pixel_dim:].set((s[:, self.pixel_dim:] - self.env_normalizer.mean) / np.sqrt(self.env_normalizer.var + self.rms_epsilon))
        return (s - self.env_normalizer.mean) / np.sqrt(self.env_normalizer.var + self.rms_epsilon)


    @partial(jax.jit, static_argnums=0)
    def input_preprocess(self, s: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([s, z], axis=-1)


    @partial(jax.jit, static_argnums=(0,))
    def _skill_preprocess(self, delta):
        if self.args.discrete:
            z_skill= jnp.eye(self.option_dim)[jnp.argmax(delta, axis=-1)]
        else:
            z_skill = delta / (jnp.linalg.norm((delta), axis=-1, keepdims=True) + 1e-8)
        return z_skill


    @partial(jax.jit, static_argnums=(0,))
    def _reward_function(self, delta, z):
        if self.args.discrete:
            maks = (z - z.mean(axis=-1, keepdims=True)) * self.option_dim / (self.option_dim - 1 if self.option_dim != 1 else 1)
            r = (delta * maks).sum(axis=-1)
        else:
            r = (delta * z).sum(axis=-1)
        return r


    @partial(jax.jit, static_argnums=(0,))
    def _option_relabel(self, target_phi_param, local_phi_params, jax_batch, relab_key, dir_weight):
        s = jax_batch['obs']
        s2 = jax_batch['next_obs']
        a = jax_batch['act']
        sc = jax_batch['after_c_obs']
        s_init = jax_batch['ep_init_obs']
        s_last = jax_batch['ep_last_obs']
        buffer_option = jax_batch['options']

        B = buffer_option.shape[0]
        ps = self.phi.apply(target_phi_param, s)
        psc = self.phi.apply(target_phi_param, sc)
        p_init = self.phi.apply(target_phi_param, s_init)
        p_last = self.phi.apply(target_phi_param, s_last)

        random_z = jax.random.normal(relab_key, (B, self.option_dim), dtype=jnp.float32)
        random_z = self._skill_preprocess(random_z)

        delta_c = psc - ps
        c_z = self._skill_preprocess(delta_c)
        delta_il = p_last - p_init
        il_z = self._skill_preprocess(delta_il)
        
        perm = jax.random.permutation(relab_key, B)
        mask_for_ac = jnp.ones((B, 1), dtype=bool).at[perm[:B//2]].set(False)
        z_ac = jnp.where(mask_for_ac, il_z, c_z)

        mask = jnp.where(dir_weight > 0.6 , True, False)
        z_ac = jnp.where(mask[:, None], buffer_option, z_ac)

        return il_z, z_ac



    @partial(jax.jit, static_argnums=(0,))
    def _direction_penalty_weight(self, disc_param, buffer_z, eps=1e-6, beta=0.2):    
        """
        return fake probability
        """
        z_novelty = self.z_count.apply(disc_param, preprocess_lsh(self.lsh_H, buffer_z))
        intrinsic_weight = jax.nn.sigmoid(z_novelty) # 0 ~ 1
        intrinsic_weight = 1 - jnp.clip(intrinsic_weight, eps, 1.0 - eps)
        return intrinsic_weight.squeeze()
    

    def learn(self, jax_batch, train_itr, train_only_local_encoder=False):
        # update local encoder
        loc_metrics = {}
        self.rng, update_rng = jax.random.split(self.rng)
        if self.args.with_local_state and self.args.decoupled_local_encoder:
            opt_states = (self.opt_global_decoder_state,
                        self.opt_local_encoder_state, 
                        self.opt_local_loglambda_state,
                        self.opt_local_decoder_log_state,
                        self.opt_local_decoder_state, 
                        self.opt_dynamics_model_state)
            params = (self.phi_params,
                    self.global_decoder_params,
                    self.local_encoder_params, 
                    self.local_encoder_loglambda,
                    self.local_decoder_log_params,
                    self.local_decoder_params, 
                    self.dynamics_model_params)
        
            new_params, new_opt_states, loc_metrics = self.update_local_encoder(jax_batch, params, opt_states, update_rng)
            self.opt_global_decoder_state, self.opt_local_encoder_state, self.opt_local_loglambda_state, self.opt_local_decoder_log_state, self.opt_local_decoder_state, self.opt_dynamics_model_state = new_opt_states
            self.global_decoder_params, self.local_encoder_params, self.local_encoder_loglambda, self.local_decoder_log_params, self.local_decoder_params, self.dynamics_model_params = new_params

            if train_only_local_encoder:
                return loc_metrics

        opt_states = (self.opt_phi_state, self.opt_actor_state,
                    self.opt_critic_state, self.opt_alpha_state, self.opt_lambda_state, self.opt_neg_lambda_state,
                    self.opt_local_encoder_state, self.opt_local_loglambda_state,
                    self.opt_local_decoder_state, self.opt_dynamics_model_state)
        params = (self.phi_params, self.target_phi_params, self.actor_params,
                self.critic_params, self.critic_target_params, self.log_alpha, self.log_lambda, self.log_neg_lambda,
                self.local_encoder_params, self.local_encoder_loglambda,
                self.target_local_encoder_params, self.local_decoder_params, self.dynamics_model_params)
        

        dir_weight = self._direction_penalty_weight(self.z_count_params, jax_batch['options'])

        if self.args.self_predictive_phi:
            update_rng, relabel_key = jax.random.split(update_rng)
            jax_batch['option_for_phi'], jax_batch['option_for_ac'] = self._option_relabel(self.target_phi_params, self.local_phi_params,
                                                        jax_batch, relabel_key,
                                                        dir_weight)
        else:
            jax_batch['option_for_phi'] = jax_batch['option_for_ac'] = jax_batch['options']

        new_params, new_opt_states, metrics, new_rng = self.update(
            jax_batch, params, opt_states, self.rng)
        
        self.opt_phi_state, self.opt_actor_state, \
        self.opt_critic_state, self.opt_alpha_state, self.opt_lambda_state, self.opt_neg_lambda_state, \
        self.opt_local_encoder_state, self.opt_local_loglambda_state, \
        self.opt_local_decoder_state, self.opt_dynamics_model_state = new_opt_states

        self.phi_params, self.target_phi_params, self.actor_params, \
        self.critic_params, self.critic_target_params, self.log_alpha, self.log_lambda, self.log_neg_lambda, \
        self.local_encoder_params, self.local_encoder_loglambda, \
        self.target_local_encoder_params, self.local_decoder_params, self.dynamics_model_params = new_params

        self.z_count_params, self.opt_z_count_state, bce_loss = self.update_count( 
            jax_batch, self.z_count_params, self.opt_z_count_state, new_rng)
        metrics['count_bce_loss'] = bce_loss

        self.rng = new_rng
        metrics.update(loc_metrics)
        return metrics
    

    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, actor_param, local_encoder_params, s, z, key):
        if self.args.with_local_state:
            s, _ = self.local_encoder.apply(local_encoder_params, s)
        a, _ = self.actor.apply(actor_param, self.input_preprocess(s, z), key)
        return a[0]


    @partial(jax.jit, static_argnums=(0,))
    def _get_eval_action(self, actor_param, local_encoder_params, s, z):
        if self.args.with_local_state:
            s, _ = self.local_encoder.apply(local_encoder_params, s)
        a = self.actor.apply(actor_param, self.input_preprocess(s, z), method='eval_action')
        return a[0]
        

    def get_action(self, s, z, eval=False):
        s = jnp.asarray(s[None]).astype(jnp.float32)
        z = jnp.asarray(z[None]).astype(jnp.float32)
        if eval:
            return np.array(self._get_eval_action(self.actor_params, self.local_encoder_params, s, z))

        self.rng, key = jax.random.split(self.rng)
        return np.array(self._get_action(self.actor_params, self.local_encoder_params, s, z, key))


    @partial(jax.jit, static_argnums=0)
    def update(self, batch, params, opt_states, rng):
        (opt_phi_state, opt_actor_state, opt_critic_state, opt_alpha_state, opt_lambda_state, opt_neg_lambda_state,
            opt_local_encoder_state, opt_local_loglambda_state, opt_local_decoder_state, opt_dynamics_model_state) = opt_states
        (phi_params, target_phi_params, actor_params, critic_params, critic_target_params, log_alpha, log_lambda, log_neg_lambda,
            local_encoder_params, local_loglambda_params, target_local_encoder_params, local_decoder_params, dynamics_model_params) = params

        rng, key, key2, key_rand, key_rand2, key_prior, subkey = jax.random.split(rng, 7)
        metrics = {}
        s = batch['obs']
        s2 = batch['next_obs']
        a = batch['act']
        done = batch['done']
        z = zphi = batch['options']
        z2 = batch['next_options']
        if self.args.self_predictive_phi:
            zphi = batch['option_for_phi']
            z = z2 = batch['option_for_ac']

        # preprocess obs
        sz = self.input_preprocess(s, z)
        s2z2 = self.input_preprocess(s2, z2)

        if self.args.with_local_state:
            ls, _ = self.local_encoder.apply(local_encoder_params, s)
            ls2, _ = self.local_encoder.apply(local_encoder_params, s2)
            lsz = self.input_preprocess(ls, z)
            ls2z2 = self.input_preprocess(ls2, z2)
            a2, logp2 = self.actor.apply(actor_params, ls2z2, key2)

            if self.args.local_for_critic:
                sz = lsz
                s2z2 = ls2z2
        else:
            a2, logp2 = self.actor.apply(actor_params, s2z2, key2)
        alpha_v = get_alpha(log_alpha)
        
        
        def phi_loss_fn(phi_p):
            ps = self.phi.apply(phi_p, s)
            ps2 = self.phi.apply(phi_p, s2)

            neg_loss = 0.0
            aug_loss = 0.0
            if self.args.self_predictive_phi:
                p_init = self.phi.apply(phi_p, batch['ep_init_obs'])
                p_last = self.phi.apply(phi_p, batch['ep_last_obs'])
                delta_il = p_last - p_init
                il_z = self._skill_preprocess(delta_il)

                eps_logits = jnp.einsum('bd,cd->bc', il_z, il_z)

                B = il_z.shape[0]
                mask = ~jnp.eye(B, dtype=bool) 
                masked_logits = jnp.where(mask, eps_logits, -jnp.inf)

                # negative term
                neg_loss = jax.nn.logsumexp(masked_logits, axis=-1).mean()
                
            delta = ps2 - ps
            dot = self._reward_function(delta, zphi)

            lam = get_lambda(log_lambda)
            lam_neg = get_lambda(log_neg_lambda)
            regul = jnp.square(ps2 - ps).mean(axis=1)
            violation = (1 - regul).clip(max=self.dual_slack)

            loss = dot + lam * violation - neg_loss# * lam_neg
            return -loss.mean(), (dot.mean(), regul.mean(), lam.mean(), aug_loss, violation, neg_loss)

        def lambda_loss_fn(log_lam, violation):
            loss = get_lambda(log_lam) * violation.mean()
            return loss

        (phi_loss, (dot, regul, lam, aug, violation_lambda, neg_loss)), grads_phi = jax.value_and_grad(
            phi_loss_fn, argnums=0, has_aux=True)(phi_params)

        grads_lam = jax.grad(lambda_loss_fn, argnums=(0))(log_lambda, violation_lambda)

        updates_phi, opt_phi_state = self.opt_phi.update(grads_phi, opt_phi_state)
        updates_lam, opt_lambda_state = self.opt_lambda.update(grads_lam, opt_lambda_state)
        phi_params = optax.apply_updates(phi_params, updates_phi)
        log_lambda = optax.apply_updates(log_lambda, updates_lam)
        
        if self.args.self_predictive_phi:
            target_neg = jnp.log(z.shape[0] - 1) + 1 / (2 * z.shape[1]) 
            violation = target_neg - neg_loss
            violation = jnp.maximum(violation, 0.0) - jnp.maximum(-violation - 0.001, 0.0)
            grads_neg_lam = jax.grad(lambda_loss_fn, argnums=(0))(log_neg_lambda, violation) 
            updates_neg_lam, opt_neg_lambda_state = self.opt_lambda.update(grads_neg_lam, opt_neg_lambda_state)
            log_neg_lambda = optax.apply_updates(log_neg_lambda, updates_neg_lam)

        # critic target
        s_phi = self.phi.apply(phi_params, s)
        s2_phi = self.phi.apply(phi_params, s2)
        delta_phi = s2_phi - s_phi
        pure_r = self._reward_function(delta_phi, z)
        r = pure_r

        q = self.critic.apply(critic_target_params, s2z2, a2).min(axis=0) - alpha_v * logp2.squeeze() # [n_critics, B, F]
        backup = r + (1. - done) * self.gamma * q  # [B, F]

        # critic
        def critic_loss_fn(c_p):
            q = self.critic.apply(c_p, sz, a)
            loss = 0.5 * ((q - backup)**2).mean()
            return loss, loss
        
        grads_c, q_loss = jax.grad(critic_loss_fn, has_aux=True)(critic_params)
        # local_loss_dict = {}
        updates_c, opt_critic_state = self.opt_critic.update(grads_c, opt_critic_state)
        critic_params = optax.apply_updates(critic_params, updates_c)
       
        # actor + alpha
        def actor_loss_fn(actor_p, log_a):
            if self.args.with_local_state:
                a_pred, logp = self.actor.apply(actor_p, lsz, key)
            else:
                a_pred, logp = self.actor.apply(actor_p, sz, key)

            q = self.critic.apply(critic_params, sz, a_pred).min(axis=0)
            alpha = get_alpha(log_a)
            loss = (alpha * logp.squeeze() - q).mean() 
            return loss, (alpha, q.mean(), logp.mean())


        def alpha_loss_fn(actor_p, log_a, logp):
            alpha = get_alpha(log_a)
            loss = -alpha * (logp + self.target_entropy).mean()
            return loss

        (actor_loss, (alpha, actor_q_min, logp)), (grads_actor,) = jax.value_and_grad(actor_loss_fn, argnums=(0,), has_aux=True)(
                                                                                                        actor_params, log_alpha)
        local_loss_dict = {}
            
        grads_alpha = jax.grad(alpha_loss_fn, argnums=(1))(actor_params, log_alpha, logp)

        updates_a, opt_actor_state = self.opt_actor.update(grads_actor, opt_actor_state)
        updates_alpha, opt_alpha_state = self.opt_alpha.update(grads_alpha, opt_alpha_state)
        actor_params = optax.apply_updates(actor_params, updates_a)
        log_alpha = optax.apply_updates(log_alpha, updates_alpha)

        # soft updates
        if self.args.self_predictive_phi:
            target_phi_params = optax.incremental_update(phi_params, target_phi_params, self.tau)

        critic_target_params = optax.incremental_update(critic_params, critic_target_params, self.tau)

        metrics.update({
            'critic_loss': q_loss,
            'phi_dot': dot, 'phi_regul': regul, 'lambda': lam, 'phi_aug': aug, 'phi_neg_loss': neg_loss, 'neg_lambda': get_lambda(log_neg_lambda).mean(),
            'actor_loss': actor_loss, 'alpha': alpha, 'actor_q_min': actor_q_min, 'actor_logp': logp, 'pure_rmean': pure_r.mean(), 'pure_rstd': pure_r.std(), 'rmean': r.mean(), 'rstd': r.std(),
        })
        metrics.update(local_loss_dict)

        new_states = (opt_phi_state, opt_actor_state,
                      opt_critic_state, opt_alpha_state, opt_lambda_state, opt_neg_lambda_state,
                      opt_local_encoder_state, opt_local_loglambda_state,
                      opt_local_decoder_state, opt_dynamics_model_state)
        new_params = (phi_params, target_phi_params, actor_params,
                critic_params, critic_target_params, log_alpha, log_lambda, log_neg_lambda,
                local_encoder_params, local_loglambda_params,
                target_local_encoder_params, local_decoder_params, dynamics_model_params)


        return new_params, new_states, metrics, rng


    @partial(jax.jit, static_argnums=(0,))
    def update_local_encoder(self, batch, params, opt_states, key):
        phi_params, global_decoder_params, local_encoder_params, local_loglambda_params, local_decoder_log_params, local_decoder_params, dynamics_model_params = params
        opt_gd_state, opt_le_state, opt_ll_state, opt_lld_state, opt_ld_state, opt_dm_state = opt_states
        
        s       = batch['obs']
        s_phi = self.phi.apply(phi_params, s)
        if self.args.obs_type == 'pixels':
                s_target = s / 255.
        else:
            s_target = s
                
        def compute_local_encoder_loss(le_params, dm_params, dec_params, beta_vib=1., gamma_dyn=1.0):
            # Unpack batch
            loc_mu, loc_logstd, loc_x = self.local_encoder.apply(le_params, s, key, method='sample')

            # VIB KL Loss
            logvar = 2.0 * loc_logstd
            var    = jnp.exp(logvar)
            
            kl_per_ex = 0.5 * jnp.sum(var + loc_mu**2 - 1.0 - logvar, axis=-1)
            kl_loss = kl_per_ex.mean()

            # Decoder Loss

            s_hat = self.local_decoder.apply(dec_params, self.input_preprocess(s_phi, loc_x))
            recon_loss = jnp.sum((s_hat - s_target)**2, axis=-1).mean()

            # Total weighted loss
            ll_lambda = get_lambda(local_loglambda_params)
            total_loss = recon_loss + kl_loss * self.local_kl_coeff

            return total_loss, ({
                "loc_recon_loss": recon_loss,
                "loc_mu_mean": loc_mu.mean(),
                "loc_kl_loss": kl_loss,
                "loc_lambda": ll_lambda,
            })

        def compute_global_decoder_loss(gd_params):            # Unpack batch
            glob_s_hat = self.global_decoder.apply(gd_params, s_phi)
            recon_loss = jnp.sum((glob_s_hat - s_target)**2, axis=-1).mean()   # → (obs_dim,)
            total_loss = recon_loss

            return total_loss, ({
                "glob_recon_loss": recon_loss,
            })


        def compute_local_decoder_loss(ld_params):            # Unpack batch
            loc_mu, _, _ = self.local_encoder.apply(local_encoder_params, s, key, method='sample')
            glob_s_hat = self.local_log_decoder.apply(ld_params, loc_mu)
            recon_loss = jnp.sum((glob_s_hat - s_target)**2, axis=-1).mean()   # → (obs_dim,)
            total_loss = recon_loss

            return total_loss, ({
                "local_log_recon_loss": recon_loss,
            })

        def lambda_loss_fn(lambda_params, violation):
            loss = -lambda_params * (violation - self.local_latent_dim).mean()
            return loss

        (total_loss, (local_loss_dict)), (grads_le, grads_dm, grads_dec) = \
        jax.value_and_grad(compute_local_encoder_loss, argnums=(0, 1, 2), has_aux=True)(
            local_encoder_params,
            dynamics_model_params,
            local_decoder_params,
        )

        (_, (global_loss_dict)), grads_gd = jax.value_and_grad(compute_global_decoder_loss, argnums=(0), has_aux=True)(global_decoder_params)
        local_loss_dict.update(global_loss_dict)
        updates_gd, new_opt_gd_state = self.opt_global_decoder.update(grads_gd, opt_gd_state)
        new_global_decoder_params = optax.apply_updates(global_decoder_params, updates_gd)


        (_, (loss_dict)), grads_ld = jax.value_and_grad(compute_local_decoder_loss, argnums=(0), has_aux=True)(local_decoder_log_params)
        local_loss_dict.update(loss_dict)
        updates_ld, new_opt_lld_state = self.opt_global_decoder.update(grads_ld, opt_lld_state)
        new_local_decoder_log_params = optax.apply_updates(local_decoder_log_params, updates_ld)

        # Encoder update
        updates_le, new_opt_le_state = self.opt_local_encoder.update(grads_le, opt_le_state)
        new_local_encoder_params = optax.apply_updates(local_encoder_params, updates_le)
        
        # Local Decoder update
        updates_dec, new_opt_ld_state = self.opt_local_decoder.update(grads_dec, opt_ld_state)
        new_local_decoder_params = optax.apply_updates(local_decoder_params, updates_dec)

        new_dynamics_model_params = dynamics_model_params
        new_opt_dm_state = opt_dm_state

        new_local_loglambda_params = local_loglambda_params
        new_opt_ll_state = opt_ll_state
        new_params = (
            new_global_decoder_params,
            new_local_encoder_params,
            new_local_loglambda_params,
            new_local_decoder_log_params,
            new_local_decoder_params,
            new_dynamics_model_params,
        )
        new_opt_states = (
            new_opt_gd_state,
            new_opt_le_state,
            new_opt_ll_state,
            new_opt_lld_state,
            new_opt_ld_state,
            new_opt_dm_state,
        )
        return new_params, new_opt_states, local_loss_dict


    @partial(jax.jit, static_argnums=(0,))
    def update_count(self, batch, disc_param, opt_state, key):
        z = batch['option_for_phi']
        fake_z = jax.random.normal(key, z.shape)
        fake_z = self._skill_preprocess(fake_z)

        z = preprocess_lsh(self.lsh_H, z)
        fake_z = preprocess_lsh(self.lsh_H, fake_z)

        x = jnp.concatenate([z, fake_z], axis=0)
        y = jnp.concatenate([
            jnp.ones((z.shape[0],), dtype=jnp.float32),
            jnp.zeros((fake_z.shape[0],), dtype=jnp.float32),
        ], axis=0)
        
        def bce_with_logits(param):
            logits = self.z_count.apply(param, x)
            bce = optax.sigmoid_binary_cross_entropy(logits, y).mean()
            return bce

        bce_loss, grads = jax.value_and_grad(bce_with_logits)(disc_param)
        updates, new_opt_state = self.opt_z_count.update(grads, opt_state)
        new_param = optax.apply_updates(disc_param, updates)

        return new_param, new_opt_state, bce_loss


    def save_checkpoint(self, epoch, log_dir: str):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'rng': self.rng,
            'phi_params': self.phi_params,
            'actor_params': self.actor_params,
            'critic_params': self.critic_params,
            'critic_target_params': self.critic_target_params,
            'log_alpha': self.log_alpha,
            'log_lambda': self.log_lambda,
            'local_encoder_params': self.local_encoder_params,
            'local_decoder_params': self.local_decoder_params,
            'global_decoder_params': self.global_decoder_params,
        }

        with open(log_dir / f'{epoch}_checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[Checkpoint Saved] → {log_dir / 'checkpoint.pkl'}")


    def load_checkpoint(self, epoch, log_dir: str):
        if epoch == -1:
            path = get_latest_checkpoint(log_dir)
        else:
            path = Path(log_dir) / f'{epoch}_checkpoint.pkl'
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.rng                     = checkpoint['rng']
        self.phi_params              = checkpoint['phi_params']
        self.actor_params            = checkpoint['actor_params']
        self.critic_params           = checkpoint['critic_params']
        self.critic_target_params    = checkpoint['critic_target_params']
        self.log_alpha               = checkpoint['log_alpha']
        self.log_lambda              = checkpoint['log_lambda']
        self.local_encoder_params    = checkpoint['local_encoder_params']
        self.local_decoder_params    = checkpoint['local_decoder_params']
        self.global_decoder_params   = checkpoint['global_decoder_params']

        print(f"[Checkpoint Loaded] ← {path}")