GPU=$1
SEED=$2
LOG_NAME=$3

python main.py \
--env_name 'quadruped_run_forward_color' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'states' \
--algo 'SFMETRA' \
--max_ep_len 200 \
--obs_dim 81 \
--action_dim 12 \
--after_c_step 25 \
--local_latent_dim 256 \
--random_rollout 100 \
--option_dim 2 \
--phi_dims 2 \
--relabel_ratio 0.0 \
--traj_itrs 2 \
--train_itrs 200 \
--with_local_state \
--decoupled_local_encoder \
--self_predictive_phi \