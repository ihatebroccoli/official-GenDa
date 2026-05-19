GPU=$1
SEED=$2
LOG_NAME=$3
EPOCH=$4
LOADDIR=$5


python downstream_task.py \
--env_name 'quadruped_run_color_goal' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'pixels' \
--algo 'SAC' \
--obs_dim 64 \
--action_dim 12 \
--csf_weight_low_bound 0.01 \
--after_c_step 50 \
--local_latent_dim 2048 \
--forward_dynamics \
--traj_itrs 8 \
--batch_size 256 \
--buffer_size 300000 \
--high_level_action_freq 10 \
--random_rollout 100 \
--max_ep_len 200 \
--goal_reset_time 500 \
--goal_range 5.0 \
--train_itrs 200 \
--load_dir ${LOADDIR} \
--desired_load_epoch ${EPOCH} \
--eval_frequency 200 \
--use_her \
