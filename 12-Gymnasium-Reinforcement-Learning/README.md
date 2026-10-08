# Gymnasium Reinforcement Learning

A collection of reinforcement learning implementations using **Gymnasium** environments, progressing from classical temporal-difference control to deep reinforcement learning for both discrete and continuous action spaces.

The notebooks combine mathematical explanations, worked calculations, algorithm implementation, training diagnostics, policy evaluation, and environment visualization.

## Learning Progression

The folder follows the progression:

**Tabular TD Control → Neural Action-Value Approximation → Deep Q-Networks → Continuous Control with DDPG**

| # | Notebook | Environment | Method | Action Space | Main Focus |
|---|---|---|---|---|---|
| 01 | `01_CliffWalking_SARSA_vs_Q_Learning.ipynb` | CliffWalking | SARSA, Q-Learning | Discrete | On-policy vs. off-policy TD control with tabular action values |
| 02 | `02_CartPole_Neural_SARSA_vs_Q_Learning.ipynb` | CartPole-v1 | Neural SARSA, Neural Q-Learning | Discrete | Neural-network action-value approximation and its stability limitations |
| 03 | `03_DQN_CartPole_Custom_Reward.ipynb` | CartPole-v1 | DQN | Discrete | Experience replay, target network, and custom reward shaping |
| 04 | `04_DQN_LunarLander.ipynb` | LunarLander-v3 | DQN | Discrete | Deep Q-learning on a more complex control problem |
| 05 | `05_DDPG_LunarLander_Continuous.ipynb` | Continuous LunarLander | DDPG | Continuous | Actor-critic learning for continuous control |
| 06 | `06_DQN_MountainCar_Custom_Reward.ipynb` | MountainCar-v0 | DQN | Discrete | Momentum-based control and custom reward shaping |
| 07 | `07_DDPG_MountainCar_Continuous.ipynb` | MountainCarContinuous-v0 | DDPG | Continuous | Continuous-control DDPG, exploration, reward design, and training stability |
| 08 | `08_DDPG_Pendulum.ipynb` | Pendulum-v1 | DDPG | Continuous | Actor-critic learning and deterministic continuous control |

## Topics Covered

The notebooks cover:

- Markov decision process interaction through Gymnasium
- State, action, reward, and episode definitions
- Epsilon-greedy exploration
- SARSA and Q-learning
- On-policy and off-policy learning
- Temporal-difference targets and TD error
- Neural-network action-value approximation
- Deep Q-Networks
- Experience replay
- Target networks
- Mini-batch training
- Reward shaping
- Actor-critic methods
- Deep Deterministic Policy Gradient
- Continuous-action exploration
- Ornstein-Uhlenbeck noise
- Soft target-network updates
- Training and evaluation metrics
- Reward and convergence visualization
- Greedy/deterministic policy evaluation
- Environment video generation

## Why Gymnasium?

Gymnasium provides standardized reinforcement learning environments with consistent state, action, reward, reset, and step interfaces.

The collection includes both major control settings:

**Discrete actions**

- CliffWalking
- CartPole
- LunarLander
- MountainCar

**Continuous actions**

- LunarLander Continuous
- MountainCarContinuous
- Pendulum

This makes the folder a progression from fundamental TD control to deep reinforcement learning for increasingly complex state and action spaces.

## Notebook Structure

Each notebook is designed as a self-contained portfolio exercise and generally includes:

1. Environment and problem definition
2. State and action spaces
3. Reward interpretation
4. Algorithm equations
5. Worked numerical examples
6. Neural-network architecture when applicable
7. Training implementation
8. Reward and learning diagnostics
9. Policy evaluation
10. Visualization or recorded simulation
11. Discussion of results and algorithm behavior

## From Neural Q-Learning to DQN

The CartPole neural SARSA/Q-learning notebook demonstrates an important limitation of directly combining nonlinear function approximation with online TD updates.

Learning can become unstable because consecutive transitions are strongly correlated and the same rapidly changing value function is also used to construct bootstrap targets.

The DQN notebooks address these problems using:

- **Experience replay** to sample less-correlated transitions
- **Target networks** to provide more stable bootstrap targets
- **Mini-batch optimization** for more stable neural-network updates

The progression illustrates why deep reinforcement learning requires more than simply replacing a Q-table with a neural network.

## DQN to DDPG

DQN is designed for discrete actions because it compares a finite set of action values:

\[
a^* = \arg\max_a Q(s,a)
\]

Continuous-control problems cannot efficiently enumerate every possible action.

DDPG addresses this by learning:

- an **actor** that directly produces a continuous action,
- a **critic** that estimates the value of a state-action pair.

This enables the final notebooks to handle continuous-action environments such as MountainCarContinuous and Pendulum.

## Installation

Install the shared dependencies with:

```bash
pip install -r requirements.txt
```

Then open the notebooks with JupyterLab, Jupyter Notebook, or Google Colab.

For local execution:

```bash
jupyter lab
```

## Main Technologies

- Python
- Gymnasium
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- ImageIO
- Jupyter

## Recommended Order

The notebook numbering represents the recommended learning sequence.

Start with tabular SARSA and Q-learning in CliffWalking, then examine neural action-value approximation in CartPole. Continue to DQN to study replay memory and target networks, and finish with DDPG for continuous-control environments.
