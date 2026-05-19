GPU=$1
SEED=$2
LOG_NAME=$3

python main.py \
--env_name 'kitchen' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'pixels' \
--algo 'SFMETRA' \
--max_ep_len 50 \
--obs_dim 64 \
--action_dim 7 \
--after_c_step 25 \
--local_latent_dim 2048 \
--random_rollout 100 \
--option_dim 10 \
--phi_dims 10 \
--train_itrs 50 \
--traj_itrs 8 \
--with_local_state \
--decoupled_local_encoder \
--self_predictive_phi \