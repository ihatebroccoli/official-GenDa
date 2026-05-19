GPU=$1
SEED=$2
LOG_NAME=$3

python main.py \
--env_name 'fish' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'states' \
--algo 'MISL' \
--max_ep_len 200 \
--obs_dim 27 \
--action_dim 5 \
--csf_weight_low_bound 0.01 \
--after_c_step 25 \
--local_latent_dim 256 \
--random_rollout 100 \
--option_dim 3 \
--phi_dims 3 \
--train_itrs 200 \
--traj_itrs 2 \
--num_epochs 25000 \
--self_predictive_phi \
--with_local_state \
--decoupled_local_encoder \