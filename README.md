# Reinforcement Learning Portfolio

A structured collection of reinforcement learning implementations, experiments, and applied analyses covering foundational methods, value-based learning, deep reinforcement learning, policy-gradient methods, reward design, Gymnasium environments, and offline reinforcement learning.

The repository emphasizes algorithm understanding, mathematical intuition, manual calculations, implementation from scratch, training diagnostics, policy evaluation, and progressive development from classical reinforcement learning to modern deep and offline RL.

## Repository Topics

| # | Topic | Main Methods / Focus |
|---:|---|---|
| 01 | [Exploration and Exploitation](./01-Exploration-and-Exploitation) | Exploration vs. exploitation, epsilon-greedy action selection, action-value estimation, multi-armed bandits |
| 02 | [Markov Decision Processes](./02-Markov-Decision-Processes) | States, actions, rewards, returns, policies, transition probabilities, Markov property, Bellman equations |
| 03 | [Dynamic Programming](./03-Dynamic-Programming) | Policy evaluation, policy improvement, policy iteration, value iteration, Bellman backups |
| 04 | [Monte Carlo Methods](./04-Monte-Carlo-Methods-on-Reinforcement-Learning) | First-visit Monte Carlo, complete-return learning, prediction, control, episodic learning |
| 05 | [Temporal-Difference Learning](./05-Temporal-Difference-Learning) | TD(0), bootstrapping, TD targets, TD error, eligibility traces, Monte Carlo vs. TD |
| 06 | [Q-Learning, SARSA, and Expected SARSA](./06-Q-Learning-SARSA-E-SARSA-Algorithms) | On-policy vs. off-policy control, Q-learning, SARSA, Expected SARSA, Taxi and GridWorld experiments |
| 07 | [Value Function Approximation](./07-Value-Function-Approximation) | Feature representations, linear approximation, gradients, prediction targets, parameter updates |
| 08 | [Introduction to Deep Learning](./08-Introduction-to-Deep-Learning) | Neural networks, forward propagation, activation functions, loss functions, gradient descent, backpropagation |
| 09 | [Deep Reinforcement Learning](./09-Deep-Reinforcement-Learning) | Deep Q-Networks, replay memory, target networks, DQN training, deep value approximation |
| 10 | [Policy Gradient Algorithms](./10-Policy-Gradient-Algorithms) | Policy gradients, actor-critic methods, advantage estimation, PPO, clipped objectives |
| 11 | [Designing Effective Reward Functions](./11.%20How-to-Design-Effective-Reward-Functions) | Reward design, reward shaping, sparse vs. dense rewards, scaling, clipping, unintended behavior |
| 12 | [Gymnasium Reinforcement Learning](./12-Gymnasium-Reinforcement-Learning) | CliffWalking, CartPole, LunarLander, MountainCar, Pendulum, SARSA, Q-learning, DQN, DDPG |
| 13 | [Offline Reinforcement Learning](./13.%20Offline-Reinforcement-Learning) | Fixed datasets, offline Q-learning, CQL, BCQ, IQL, Minari, D4RL, continuous offline control |

## What This Repository Demonstrates

- Reinforcement learning fundamentals and sequential decision-making
- Exploration and exploitation strategies
- Markov Decision Processes and Bellman equations
- Dynamic programming for planning with known models
- Monte Carlo and temporal-difference learning
- On-policy and off-policy control
- Q-learning, SARSA, and Expected SARSA
- Value function approximation with features and gradients
- Neural-network foundations for deep reinforcement learning
- Deep Q-Networks with replay memory and target networks
- Policy-gradient and actor-critic methods
- Proximal Policy Optimization
- Reward-function design and reward shaping
- Discrete and continuous control with Gymnasium
- Deep Deterministic Policy Gradient
- Offline reinforcement learning from fixed datasets
- Conservative and batch-constrained offline RL
- Implicit Q-Learning
- Training diagnostics, reward curves, convergence analysis, and policy evaluation
- Environment visualization and simulation videos

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Gymnasium
- MiniGrid
- Minari
- TensorFlow / Keras
- PyTorch
- d3rlpy
- Gymnasium-Robotics
- MuJoCo
- ImageIO
- Jupyter Notebook

## Repository Organization

Each topic is organized as a self-contained section with notebooks, supporting files when required, and topic-specific README documentation describing the concepts, algorithms, implementation, experiments, and results.

```text
reinforcement-learning-portfolio/
├── 01-Exploration-and-Exploitation/
├── 02-Markov-Decision-Processes/
├── 03-Dynamic-Programming/
├── 04-Monte-Carlo-Methods-on-Reinforcement-Learning/
├── 05-Temporal-Difference-Learning/
├── 06-Q-Learning-SARSA-E-SARSA-Algorithms/
├── 07-Value-Function-Approximation/
├── 08-Introduction-to-Deep-Learning/
├── 09-Deep-Reinforcement-Learning/
├── 10-Policy-Gradient-Algorithms/
├── 11. How-to-Design-Effective-Reward-Functions/
├── 12-Gymnasium-Reinforcement-Learning/
├── 13. Offline-Reinforcement-Learning/
├── .gitignore
└── README.md
```

## Learning Progression

The repository is organized to show a progression from foundational reinforcement learning to deep and offline methods:

```text
Exploration and Exploitation
        ↓
Markov Decision Processes
        ↓
Dynamic Programming
        ↓
Monte Carlo Methods
        ↓
Temporal-Difference Learning
        ↓
Q-Learning / SARSA / Expected SARSA
        ↓
Value Function Approximation
        ↓
Deep Learning Foundations
        ↓
Deep Reinforcement Learning
        ↓
Policy Gradient Methods
        ↓
Reward Function Design
        ↓
Gymnasium Applications
        ↓
Offline Reinforcement Learning
```

## Focus

The repository shows progression from fundamental reinforcement learning concepts to value-based control, function approximation, deep reinforcement learning, policy-gradient methods, reward design, continuous control, and offline reinforcement learning.

The emphasis is on readable implementations, mathematical understanding, worked examples, reproducible experiments, training diagnostics, and clear interpretation of learned behavior and algorithm performance.
