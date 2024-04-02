# Instructions for reproducing the HW4 results

## Q1:

```
python run_hw4_gc.py \
     env.env_name=reacher \
     env.exp_name=q1_reacher \
     env.goal_dist=uniform \
     env.goal_rep=absolute \
     env.goal_indicies=[-6,-5,-4] \
     env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
     env.gaussian_bounds="[[ 0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
     alg.rl_alg=td3 \
     alg.on_policy=False \
     env.task_name=gcrl \
     alg.n_iter=200001 \
     alg.network.layer_sizes=[256,256] \
     alg.network.activations=[relu,relu] \
     alg.td3_target_policy_noise=0.1 \
     alg.td3_target_policy_noise_clip=0.5 \
     alg.num_agent_train_steps_per_iter=4 \
     alg.learning_rate=1e-4  \
     alg.critic_learning_rate=1e-3 \
     logging.video_log_freq=50000 \
     alg.batch_size=128 \
     alg.train_batch_size=128 \
     alg.gamma=0.99 \
     alg.learning_starts=6000 \
     alg.replay_buffer_size=1000000 \
     alg.learning_freq=2 \
     alg.frame_history_len=1 \
     alg.target_update_freq=1 \
     alg.polyak_avg=0.995 \
     alg.grad_norm_clipping=1.0 \
     alg.network.output_activation=tanh \
     logging.random_seed=37 \
     alg.use_gpu=True \

  seed 2
 python run_hw4_gc.py \
     env.env_name=reacher \
     env.exp_name=q1_reacher \
     env.goal_dist=uniform \
     env.goal_rep=absolute \
     env.goal_indicies=[-6,-5,-4] \
     env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
     env.gaussian_bounds="[[ 0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
     alg.rl_alg=td3 \
     alg.on_policy=False \
     env.task_name=gcrl \
     alg.n_iter=200001 \
     alg.network.layer_sizes=[256,256] \
     alg.network.activations=[relu,relu] \
     alg.td3_target_policy_noise=0.1 \
     alg.td3_target_policy_noise_clip=0.5 \
     alg.num_agent_train_steps_per_iter=4 \
     alg.learning_rate=1e-4  \
     alg.critic_learning_rate=1e-3 \
     logging.video_log_freq=50000 \
     alg.batch_size=128 \
     alg.train_batch_size=128 \
     alg.gamma=0.99 \
     alg.learning_starts=6000 \
     alg.replay_buffer_size=1000000 \
     alg.learning_freq=2 \
     alg.frame_history_len=1 \
     alg.target_update_freq=1 \
     alg.polyak_avg=0.995 \
     alg.grad_norm_clipping=1.0 \
     alg.network.output_activation=tanh \
     logging.random_seed=10237 \
     alg.use_gpu=True \


  seed 3
 python run_hw4_gc.py \
     env.env_name=reacher \
     env.exp_name=q1_reacher \
     env.goal_dist=uniform \
     env.goal_rep=absolute \
     env.goal_indicies=[-6,-5,-4] \
     env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
     env.gaussian_bounds="[[ 0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
     alg.rl_alg=td3 \
     alg.on_policy=False \
     env.task_name=gcrl \
     alg.n_iter=200001 \
     alg.network.layer_sizes=[256,256] \
     alg.network.activations=[relu,relu] \
     alg.td3_target_policy_noise=0.1 \
     alg.td3_target_policy_noise_clip=0.5 \
     alg.num_agent_train_steps_per_iter=4 \
     alg.learning_rate=1e-4  \
     alg.critic_learning_rate=1e-3 \
     logging.video_log_freq=50000 \
     alg.batch_size=128 \
     alg.train_batch_size=128 \
     alg.gamma=0.99 \
     alg.learning_starts=6000 \
     alg.replay_buffer_size=1000000 \
     alg.learning_freq=2 \
     alg.frame_history_len=1 \
     alg.target_update_freq=1 \
     alg.polyak_avg=0.995 \
     alg.grad_norm_clipping=1.0 \
     alg.network.output_activation=tanh \
     logging.random_seed=8177 \
     alg.use_gpu=True \
```

##################################################################

## Q2:

### Reacher
```
python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q2_reacher_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=10237 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q2_reacher_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=8177 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q2_reacher_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=8492 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \
```

### WidowX

#### Uniform
```
python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_uniform \
    env.goal_dist=uniform \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=62117 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_uniform \
    env.goal_dist=uniform \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=3338 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_uniform \
    env.goal_dist=uniform \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=1829 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \
```

#### Normal

```
python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=62117 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=3338 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q2_widowx_normal \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=1829 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \
```

##################################################################

## Q3:

### Reacher

```
python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q3_reacher_normal_relative \
    env.goal_dist=normal \
    env.goal_rep=relative \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[ 0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200001 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=4 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=2 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=37 \
    alg.use_gpu=True \
```

### WidowX

```
python run_hw4_gc.py \
    env.env_name=widowx \
    env.exp_name=q3_widowx_normal_relative \
    env.goal_dist=normal \
    env.goal_rep=relative \
    env.goal_indicies=[0,1,2] \
    env.uniform_bounds="[[0.4,-0.2,-0.34],[0.8,0.4,-0.1]]" \
    env.gaussian_bounds="[[0.6,0.1,-0.2],[0.2,0.2,0.2]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl \
    alg.n_iter=200005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=62117 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \
```

####################################################

## Q4:

### Reacher

```
python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q4_reacher_normal_gf5 \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[ 0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=gcrl_v2 \
    env.goal_frequency=5 \
    alg.n_iter=400005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=256 \
    alg.train_batch_size=256 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=37 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \

```

##################################################################

## Q5:

### Reacher

```
python run_hw4_gc.py \
    env.env_name=reacher \
    env.exp_name=q5_hrl_gf5 \
    env.goal_dist=normal \
    env.goal_rep=absolute \
    env.goal_indicies=[-6,-5,-4] \
    env.uniform_bounds="[[-0.6,-1.4,-0.4],[0.8,0.2,0.5]]" \
    env.gaussian_bounds="[[0.2,-0.7,0.0],[0.3,0.4,0.05]]" \
    alg.rl_alg=td3 \
    alg.on_policy=False \
    env.task_name=hrl \
    env.low_level_policy_path=/network/scratch/s/sarvjeet-singh.ghotra/git/robot_learning/data/hw4_q4_reacher_normal_gf5_reacher_02-04-2024_03-15-11/agent_itr_300000.pt \
    env.goal_frequency=5 \
    alg.n_iter=300005 \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.1 \
    alg.td3_target_policy_noise_clip=0.5 \
    alg.num_agent_train_steps_per_iter=2 \
    alg.num_critic_updates_per_agent_update=2 \
    alg.learning_rate=1e-4  \
    alg.critic_learning_rate=1e-3 \
    logging.video_log_freq=50000 \
    alg.batch_size=128 \
    alg.train_batch_size=128 \
    alg.gamma=0.99 \
    alg.learning_starts=6000 \
    alg.replay_buffer_size=1000000 \
    alg.learning_freq=1 \
    alg.frame_history_len=1 \
    alg.target_update_freq=1 \
    alg.polyak_avg=0.995 \
    alg.grad_norm_clipping=1.0 \
    alg.network.output_activation=tanh \
    logging.random_seed=91 \
    logging.save_params=True \
    logging.save_params_freq=50000 \
    logging.scalar_log_freq=4000 \
    alg.use_gpu=True \
```
