GPU=$1
SEED=$2
LOG_NAME=$3

python main.py \
--env_name 'humanoid_run_color' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'pixels' \
--algo 'SFMETRA' \
--max_ep_len 200 \
--action_dim 21 \
--csf_weight_low_bound 0.01 \
--after_c_step 25 \
--obs_dim 64 \
--pixel_latent_dim 256 \
--local_latent_dim 2048 \
--random_rollout 100 \
--option_dim 2 \
--phi_dims 2 \
--buffer_size 300000 \
--frame_stack 3 \
--batch_size 256 \
--traj_itrs 4 \
--train_itrs 200 \
--self_predictive_phi \
--with_local_state \
--decoupled_local_encoder \
