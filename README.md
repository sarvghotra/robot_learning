[![tests](https://github.com/milarobotlearningcourse/robot_learning/actions/workflows/testing.yaml/badge.svg)](https://github.com/milarobotlearningcourse/robot_learning/actions/workflows/testing.yaml)

# Setup

You can run this code on your own machine or on Google Colab (Colab is not completely supported).

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt` if you are running the assignment locally.

2. **Docker:** You can also run this code in side of a docker image. You will need to build the docker image using the provided docker file

3. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

### Anaconda

If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then create a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).

```
conda create -n roble python=3.8
conda activate roble
pip install -r requirements.txt
```

If having issues with Pytorch and GPU, make sure to install the compatible version of Pytorch for your CUDA version [here](https://pytorch.org/get-started/locally/)



## Examples:


```
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false
```

```
python run_hw6_sim2real.py alg.n_iter=1
```



Assignments for [UdeM roble: Robot Learning Course](https://fracturedplane.com/teaching-new-course-in-robot-learning.html). Based on [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).


# Instructions for HW1 experiments

## Q1-2 (BC):
- Ant-V2: `python run_hw1_bc.py --config-name config_hw1_BC1`
- Humanoid-V2: `python run_hw1_bc.py --config-name config_hw1_BC1_Humanoid_not_work`

## Q1-3:
- Batch size hyper-param sweep for Ant-V2:
    ```
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=32 env.exp_name=q1-3_32bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=64 env.exp_name=q1-3_64bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=128 env.exp_name=q1-3_128bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=256 env.exp_name=q1-3_256bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=512 env.exp_name=q1-3_512bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=1024 env.exp_name=q1-3_1024bs_ant
    python run_hw1_bc.py --config-name config_hw1_BC3_bs alg.train_batch_size=1400 env.exp_name=q1-3_1400bs_ant
    ```

## Q1.1 (IDM):
- Ant-V2: `python run_hw1_bc.py --config-name config_hw1_IDM`
- HalfCheetah-V2: `python run_hw1_bc.py --config-name config_hw1_IDM_halfcheetah`

## Q2 (DAgger):
- Ant-V2: `python run_hw1_bc.py --config-name config_hw1_DAg_ant`
- Humanoid-V2: `python run_hw1_bc.py --config-name config_hw1_DAg_Humanoid`
