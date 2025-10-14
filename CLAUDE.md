# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Dynamically Optimal Treatment Allocation** using reinforcement learning (Adusumilli, Geiecke, Schilter, 2024). The project uses Proximal Policy Optimization (PPO) to learn treatment allocation policies for job training programs, based on the JTPA (Job Training Partnership Act) dataset.

## Environment Setup

```bash
# Install miniconda if not already installed
# Create environment from YAML file
conda env create -f code/dynamictreatment.yml

# Activate environment
conda activate dynamictreatment
```

## Running the Code

Main entry point is `ppo.py` which trains PPO agents:

```bash
# Navigate to code directory
cd code

# Basic run with default parameters
python ppo.py

# Run with custom parameters (example)
python ppo.py --seed 42 --epochs 2001 --cpu 10 --steps 60000 --logistic_policy 1 --non_stationary_policy 0
```

### Key Command-Line Arguments

- `--seed`: Random seed for reproducibility
- `--cpu`: Number of CPUs for parallel MPI training (default: 10)
- `--epochs`: Number of training epochs (default: 2001)
- `--steps`: Steps per epoch across all processes (default: 60000)
- `--hid`: Neurons per hidden layer (default: 64)
- `--l`: Number of hidden layers (default: 2)
- `--logistic_policy`: Use logistic policy (1) or MLP policy (0) (default: 1)
- `--non_stationary_policy`: Include budget/time in policy (1) or not (0) (default: 0)
- `--pi_lr`: Policy learning rate (default: 0.1)
- `--vf_lr`: Value function learning rate (default: 1e-3)
- `--save_freq`: Save model every N epochs (default: 500)
- `--exp_name`: Experiment name for logging (default: "ppo")

## Architecture Overview

### Core Components

1. **`ppo.py`**: Main training script implementing PPO algorithm
   - Contains `PPOBuffer` class for trajectory storage with GAE-Lambda
   - `ppo()` function orchestrates training loop with MPI parallelization
   - Saves checkpoints to `../data/output/runs/`

2. **`core.py`**: Neural network architectures adapted from OpenAI Spinning Up
   - `ActorCritic`: Combined actor-critic module
   - `LogisticCategoricalActor`: Logistic/sigmoid policy for binary actions
   - `MLPCategoricalActor`: Multi-layer perceptron policy
   - `MLPCritic`: Value function approximator

3. **`rctenvironments/jobtraining.py`**: Custom RL environment
   - `DynamicJTPA`: Gym-style environment simulating dynamic treatment allocation
   - Loads JTPA data, estimates individual treatment effects (ITEs)
   - Implements budget constraints, time constraints, and monthly arrival rates
   - State space includes: individual covariates, remaining budget, time
   - Action space: binary (treat=1, don't treat=0)

4. **`rctenvironments/ite_estimation.py`**: Treatment effect estimation
   - `add_doubly_robust_ites()`: Computes doubly robust ITE estimates using leave-one-out cross-validation
   - Used to construct reward functions when not using pre-computed KT 2018 ITEs

5. **`utils.py`**: Utilities from OpenAI Spinning Up
   - MPI parallelization functions (`mpi_fork`, `mpi_avg_grads`, `sync_params`)
   - `EpochLogger`: Logging and checkpoint saving
   - PyTorch-MPI integration helpers

### Data Flow

1. **Data Loading**: `jobtraining.py` loads JTPA data from `../data/input/`
   - Merges KT 2018 data with JTPA National Evaluation data
   - Computes rewards (ITEs) and costs

2. **Environment**: `DynamicJTPA` environment samples individuals sequentially
   - Arrival times follow monthly exponential distributions
   - Budget depletes with treatments, time advances
   - Episode terminates when budget exhausted or time horizon reached

3. **Training**: `ppo.py` runs multiple parallel workers via MPI
   - Each worker collects trajectories in its environment
   - Gradients averaged across workers
   - Periodic checkpointing to disk

4. **Output**: Trained models and logs saved to `../data/output/runs/`

### Policy Types

The code supports two policy architectures controlled by `--logistic_policy`:

- **Logistic Policy** (`logistic_policy=1`): Single-layer logistic regression with optional time dummies for non-stationary policies
- **MLP Policy** (`logistic_policy=0`): Multi-layer perceptron with configurable hidden layers

Both can be stationary (covariates only) or non-stationary (including budget/time).

### Important Implementation Details

- **MPI Parallelization**: Training uses `mpi4py` to run multiple workers in parallel. The `mpi_fork()` function relaunches the script with `mpirun`.
- **State Normalization**: Covariates are standardized; budget and time optionally shifted to [-0.5, 0.5] range
- **Reward Scaling**: ITEs scaled to have unit standard deviation
- **Cost Scaling**: Costs scaled so budget of 1 can treat ~25% of yearly arrivals (configurable)
- **Monthly Dynamics**: Arrival rates and costs vary by month when `--monthly_arrival_rates 1`

## Data Requirements

Expected input data structure in `data/input/`:
- `jtpa_kt.tab`: Kitagawa-Tetenov 2018 data
- `expbif.dta`: JTPA National Evaluation data
- `ites_kt.csv`: Pre-computed ITEs (optional, only if `--kt_rewards 1`)

Output structure:
- `data/output/runs/`: Training logs and model checkpoints
- `data/output/environment/`: Processed environment data (optional)
