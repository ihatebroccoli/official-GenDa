GPU=$1
SEED=$2
LOG_NAME=$3
EPOCH=$4
LOADDIR=$5


python downstream_task.py \
--env_name 'humanoid_run_maze_goal' \
--device ${GPU} \
--seed ${SEED} \
--exp_tag ${LOG_NAME} \
--obs_type  'states' \
--algo 'SAC' \
--obs_dim 70 \
--action_dim 21 \
--csf_weight_low_bound 0.01 \
--after_c_step 50 \
--local_latent_dim 2048 \
--forward_dynamics \
--traj_itrs 8 \
--batch_size 256 \
--buffer_size 1000000 \
--high_level_action_freq 25 \
--random_rollout 100 \
--train_itrs 200 \
--load_dir ${LOADDIR} \
--desired_load_epoch ${EPOCH} \
--eval_frequency 200 \
--num_epochs 25000 \
--use_her \
--max_ep_len 500 \
--goal_reset_time 600 \
--map_name H \

# 50000
# L: 400
# H: 500