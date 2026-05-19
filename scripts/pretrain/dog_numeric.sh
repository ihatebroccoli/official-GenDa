GPU=$1
SEED=$2
LOG_NAME=$3

python main.py \
--env_name 'dog_run' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'states' \
--algo 'MISL' \
--max_ep_len 200 \
--obs_dim 226 \
--action_dim 38 \
--csf_weight_low_bound 0.01 \
--after_c_step 25 \
--local_latent_dim 512 \
--random_rollout 100 \
--option_dim 2 \
--phi_dims 2 \
--train_itrs 200 \
--traj_itrs 2 \
--self_predictive_phi \
--with_local_state \
--decoupled_local_encoder \