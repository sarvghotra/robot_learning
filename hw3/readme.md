# Instructions to run the experiments:

## Q1:
```Python
python run_hw3_ql.py \
    "env.exp_name=q1" \
    "env.env_name=MsPacman-v0" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=8000000" \
    "logging.random_seed=2711" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-3" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=50000" \
    "alg.learning_freq=4" \
    "alg.frame_history_len=4" \
    "alg.target_update_freq=10000" \
    "alg.grad_norm_clipping=2.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=20000" \
    "alg.eval_batch_size=1000"
```

## Q2:

### DQN
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_dqn_1" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=1" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_dqn_1" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=2" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_dqn_1" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=3" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```

## DDPG:
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_doubledqn_1" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=True" \
    "alg.n_iter=500000" \
    "logging.random_seed=1" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_doubledqn_2" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=True" \
    "alg.n_iter=500000" \
    "logging.random_seed=2" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```
```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q2_doubledqn_3" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=True" \
    "alg.n_iter=500000" \
    "logging.random_seed=3" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```

## Q3:

```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q3_hparam1" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=1" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```

```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q3_hparam2" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=1" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=4" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```

```Python
python run_hw3_ql_dqn.py \
    "env.exp_name=q3_hparam3" \
    "env.env_name=LunarLander-v3" \
    "alg.rl_alg=dqn" \
    "alg.double_q=False" \
    "alg.n_iter=500000" \
    "logging.random_seed=1" \
    "alg.replay_buffer_size=100000" \
    "alg.num_agent_train_steps_per_iter=6" \
    "alg.learning_rate=1e-4" \
    "alg.batch_size=64" \
    "alg.train_batch_size=64" \
    "alg.gamma=0.99" \
    "alg.learning_starts=2000" \
    "alg.learning_freq=1" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=3000" \
    "alg.grad_norm_clipping=1.0" \
    "logging.video_log_freq=50000" \
    "logging.scalar_log_freq=10000" \
```

## Q4:

```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up4_lr1e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=4" \
    "alg.learning_rate=1e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up4_lr5e-5" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=4" \
    "alg.learning_rate=5e-5" \
    "alg.critic_learning_rate=5e-4" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up4_lr2e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=4" \
    "alg.learning_rate=2e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=40000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```


```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up1_lr2e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=2e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up2_lr2e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=2e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q4_ddpg_up8_lr2e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=8" \
    "alg.learning_rate=2e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

## Q5:

```Python
python run_hw3_ql.py \
    "env.env_name=HalfCheetah-v2" \
    "env.exp_name=q5_ddpg_hard_up1_lr2e-4" \
    "alg.rl_alg=ddpg" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.num_agent_train_steps_per_iter=1" \
    "alg.learning_rate=2e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=100" \
    "alg.polyak_avg=0.9" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=80000" \
    "logging.random_seed=37" \
    "alg.network.output_activation=tanh" \
    "env.max_episode_length=1000"
```

## TD3

## Q6:

```Python
python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_256-256_rho0.05" \
    alg.rl_alg=td3 \
    env.atari=false \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.05 \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000


python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_256-256_rho0.1" \
    alg.rl_alg=td3 \
    env.atari=false \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000


python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_256-256_rho0.2" \
    alg.rl_alg=td3 \
    env.atari=false \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.2 \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000
```

```Python
python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_64-128-64_rho0.05" \
    alg.rl_alg=td3 \
    env.atari=false \
    alg.network.layer_sizes=[64,128,64] \
    alg.network.activations=[relu,relu,relu] \
    alg.td3_target_policy_noise=0.05 \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000
```

```Python
python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_256-256_rho0.05" \
    alg.rl_alg=td3 \
    env.atari=false \
    alg.network.layer_sizes=[256,256] \
    alg.network.activations=[relu,relu] \
    alg.td3_target_policy_noise=0.05 \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000
```

```Python
python run_hw3_ql.py \
    env.env_name="InvertedPendulum-v2" \
    env.exp_name="q6_td3_shape_128-128-128_rho0.05" \
    alg.rl_alg=td3 \
    env.atari=false \
    alg.network.layer_sizes=[64,128,64] \
    alg.network.activations=[relu,relu,relu] \
    alg.td3_target_policy_noise=0.05 \
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
    alg.n_iter=60000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000
```

## Q7:

```Python
python run_hw3_ql.py \
    env.env_name="HalfCheetah-v2" \
    env.exp_name="q7_td3_shape_256-256_rho0.1" \
    alg.rl_alg=td3 \
    env.atari=false \
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
    alg.n_iter=80000 \
    logging.random_seed=37 \
    alg.network.output_activation=tanh \
    env.max_episode_length=1000
```

## SAC

## Q8:
```Python
python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q8_sac_alpha0.05" \
    "alg.rl_alg=sac" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.entropy_coeff=0.05" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=1e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=1" \
    "alg.polyak_avg=0.995" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "env.max_episode_length=1000"


python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q8_sac_alpha0.01" \
    "alg.rl_alg=sac" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.entropy_coeff=0.01" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=1e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=1" \
    "alg.polyak_avg=0.995" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "env.max_episode_length=1000"


python run_hw3_ql.py \
    "env.env_name=InvertedPendulum-v2" \
    "env.exp_name=q8_sac_alpha0.1" \
    "alg.rl_alg=sac" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.entropy_coeff=0.1" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=1e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=1" \
    "alg.polyak_avg=0.995" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=60000" \
    "logging.random_seed=37" \
    "env.max_episode_length=1000"
```

## Q9:

```Python
python run_hw3_ql.py \
    "env.env_name=HalfCheetah-v2" \
    "env.exp_name=q8_sac_alpha0.05" \
    "alg.rl_alg=sac" \
    "env.atari=false" \
    "alg.discrete=False" \
    "alg.entropy_coeff=0.05" \
    "alg.num_agent_train_steps_per_iter=2" \
    "alg.learning_rate=1e-4" \
    "alg.critic_learning_rate=1e-3" \
    "logging.video_log_freq=20000" \
    "alg.batch_size=128" \
    "alg.train_batch_size=128" \
    "alg.gamma=0.99" \
    "alg.learning_starts=6000" \
    "alg.replay_buffer_size=1000000" \
    "alg.learning_freq=2" \
    "alg.frame_history_len=1" \
    "alg.target_update_freq=1" \
    "alg.polyak_avg=0.995" \
    "alg.grad_norm_clipping=1.0" \
    "alg.n_iter=80000" \
    "logging.random_seed=37" \
    "env.max_episode_length=1000"
```