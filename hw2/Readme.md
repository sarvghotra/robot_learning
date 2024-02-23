============== Question 1 ==============

```Python
python run_hw2_mb.py \
    env.exp_name=q1_cheetah_n500_arch1x32 \
    env.env_name=cheetah-roble-v0 \
    alg.num_agent_train_steps_per_iter=500 \
    alg.network.layer_sizes=[32] \
    alg.n_iter=1
```

```Python
python run_hw2_mb.py \
    env.exp_name=q1_cheetah_n5_arch2x256 \
    env.env_name=cheetah-roble-v0 \
    alg.num_agent_train_steps_per_iter=5 \
    alg.network.layer_sizes=[256,256] \
    alg.n_iter=1
```

```Python
python run_hw2_mb.py \
    env.exp_name=q1_cheetah_n500_arch2x256 \
    env.env_name=cheetah-roble-v0 \
    alg.num_agent_train_steps_per_iter=500 \
    alg.network.layer_sizes=[256,256] \
    alg.n_iter=1
```

============== Question 2 ==============
```Python
python run_hw2_mb.py \
    env.exp_name=q2_obstacles_singleiteration \
    env.env_name=obstacles-roble-v0 \
    alg.num_agent_train_steps_per_iter=20 \
    alg.batch_size_initial=5000 \
    alg.batch_size=1000 \
    alg.mpc_horizon=10 \
    alg.n_iter=1 \
    logging.video_log_freq=2
```

============== Question 3 ==============
```Python
python run_hw2_mb.py \
    env.exp_name=q3_obstacles \
    env.env_name=obstacles-roble-v0 \
    alg.num_agent_train_steps_per_iter=20 \
    alg.batch_size_initial=5000 \
    alg.batch_size=1000 \
    alg.mpc_horizon=10 \
    alg.n_iter=12 \
    logging.video_log_freq=3 \
    alg.seed=52 \
    logging.random_seed=180
```

```Python
python run_hw2_mb.py \
    env.exp_name=q3_reacher \
    env.env_name=reacher-roble-v0 \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size_initial=5000 \
    alg.batch_size=5000 \
    alg.n_iter=15 \
    logging.video_log_freq=2
```

```Python
python run_hw2_mb.py \
    env.exp_name=q3_cheetah \
    env.env_name=cheetah-roble-v0 \
    alg.mpc_horizon=15 \
    alg.num_agent_train_steps_per_iter=1500 \
    alg.batch_size_initial=5000 \
    alg.batch_size=5000 \
    alg.n_iter=20 \
    logging.video_log_freq=2
```

============== Question 4 ==============
```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_horizon5 \
    env.env_name=reacher-roble-v0 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=5 \
    alg.mpc_action_sampling_strategy='random' \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_horizon15 \
    env.env_name=reacher-roble-v0 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=15 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_horizon30 \
    env.env_name=reacher-roble-v0 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=30 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_numseq100 \
    env.env_name=reacher-roble-v0 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    alg.mpc_num_action_sequences=100 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_numseq1000 \
    env.env_name=reacher-roble-v0 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_num_action_sequences=1000 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_ensemble1 \
    env.env_name=reacher-roble-v0 \
    alg.ensemble_size=1 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_ensemble3 \
    env.env_name=reacher-roble-v0 \
    alg.ensemble_size=3 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q4_reacher_ensemble5 \
    env.env_name=reacher-roble-v0 \
    alg.ensemble_size=5 \
    alg.add_sl_noise=true \
    alg.mpc_horizon=10 \
    alg.num_agent_train_steps_per_iter=1000 \
    alg.batch_size=800 \
    alg.n_iter=15 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

=============== Question 5 ====================
```Python
python run_hw2_mb.py \
    env.exp_name=q5_cheetah_random \
    env.env_name='cheetah-roble-v0' \
    alg.mpc_horizon=15 \
    alg.num_agent_train_steps_per_iter=1500 \
    alg.batch_size_initial=5000 \
    alg.batch_size=5000 \
    alg.n_iter=5 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='random'
```

```Python
python run_hw2_mb.py \
    env.exp_name=q5_cheetah_cem_2 \
    env.env_name='cheetah-roble-v0' \
    alg.mpc_horizon=15 \
    alg.add_sl_noise=true \
    alg.num_agent_train_steps_per_iter=1500 \
    alg.batch_size_initial=5000 \
    alg.batch_size=5000 \
    alg.n_iter=5 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='cem' \
    alg.cem_iterations=2
```

```Python
python run_hw2_mb.py \
    env.exp_name=q5_cheetah_cem_4 \
    env.env_name='cheetah-roble-v0' \
    alg.mpc_horizon=15 \
    alg.add_sl_noise=true \
    alg.num_agent_train_steps_per_iter=1500 \
    alg.batch_size_initial=5000 \
    alg.batch_size=5000 \
    alg.n_iter=5 \
    logging.video_log_freq=2 \
    alg.mpc_action_sampling_strategy='cem' \
    alg.cem_iterations=4 \
    alg.seed=1839337
```
