# 06 — Q-Learning, SARSA, and Expected SARSA Algorithms

This folder contains a reinforcement learning portfolio focused on **tabular TD control**, beginning with Q-Learning, SARSA, and Expected SARSA and then applying Q-Learning to increasingly challenging environments.

The notebooks progress from small tabular examples to robot navigation, multi-agent learning, standard Gymnasium environments, continuous-state discretization, and a high-dimensional Atari example.

## Algorithms

### Q-Learning

Q-Learning is an off-policy TD control algorithm:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma\max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right].
$$

### SARSA

SARSA is an on-policy TD control algorithm:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
-
Q(S_t,A_t)
\right].
$$

### Expected SARSA

Expected SARSA replaces the single sampled next action with the expected next-state action value:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\sum_a
\pi(a|S_{t+1})Q(S_{t+1},a)
-
Q(S_t,A_t)
\right].
$$

## Notebook Order

| # | Notebook | Main focus |
|---|---|---|
| 01 | `01_Q_Learning_SARSA_E_SARSA_Algorithms.ipynb` | RL recap, Q-Learning foundations, six-state assignment, SARSA, Expected SARSA, on-policy vs. off-policy learning, and Taxi comparison |
| 02 | `02_Q_Learning_Indoor_Robot_Path_Navigation_Paper_Reproduction.ipynb` | Q-Learning reproduction of an indoor robot path-navigation problem, obstacle avoidance, learned paths, Q-value plots, and real-robot workflow |
| 03 | `03_Multi_Agent_Q_Learning_Grid_Navigation.ipynb` | Multi-agent RL concepts, independent Q-Learning, Markov-game formulation, individual Q-tables, convergence, coordination, collisions, and step-by-step two-agent simulation |
| 04 | `04_Q_Learning_FrozenLake_OpenAI_Gymnasium.ipynb` | FrozenLake-v1, exact optimal Q-values, learned Q-table, convergence, greedy policy, and trained-agent rendering |
| 05 | `05_Q_Learning_Taxi_OpenAI_Gymnasium.ipynb` | Taxi, encoded states, exact optimal Q-values, learned Q-table, convergence, and trained-agent simulation |
| 06 | `06_Q_Learning_CliffWalking_OpenAI_Gymnasium.ipynb` | CliffWalking-v1, cliff penalties, exact optimal Q-values, learned policy, convergence, and rendering |
| 07 | `07_Q_Learning_Blackjack_OpenAI_Gymnasium.ipynb` | Blackjack-v1, reference optimal Q-values, training, convergence, policy maps, and evaluation |
| 08 | `08_Q_Learning_MountainCar_OpenAI_Gymnasium.ipynb` | MountainCar-v0, continuous-state discretization, tabular Q-Learning, success rate, Q-value maps, and animation |
| 09 | `09_Q_Learning_MarioBros_18_Actions_Gymnasium_ALE.ipynb` | ALE/MarioBros-v5, 18 discrete actions, RAM-state aggregation, approximate tabular Q-Learning, Q-table growth, evaluation, and rendering |

## What Is Covered

Across the notebooks, the portfolio includes:

- reinforcement learning components: state, action, reward, policy, value function, and model;
- exploration vs. exploitation and $\epsilon$-greedy action selection;
- TD targets and TD errors;
- Q-Learning, SARSA, and Expected SARSA;
- on-policy vs. off-policy learning;
- behavior and target policies;
- Q-table initialization, training, and convergence;
- first 60 Q-value updates in several problems;
- manual numerical Q-value calculations;
- learned greedy policies;
- reward and episode-length graphs;
- exact/reference $Q^*$ comparisons where practical;
- obstacle-based robot path navigation;
- multi-agent Q-Learning and Markov-game concepts;
- individual agent rewards, Q-tables, convergence curves, collisions, and joint success;
- step-by-step multi-agent simulation;
- continuous-state discretization for MountainCar;
- high-dimensional state aggregation for Atari/MarioBros;
- rendered trained-agent simulations and GIFs.

## Multi-Agent Notebook

The multi-agent notebook is an **educational reconstruction inspired by the lecture slide** on multi-agent Q-Learning. The slide does not define the exact number of agents, state representation, reward structure, or learning architecture, so those details are explicitly designed in the notebook for teaching purposes.

For two agents, the global state is represented by both positions:

$$
S_t=(p_t^1,p_t^2).
$$

Each agent uses its own ordered observation:

$$
o_t^1=(p_t^1,p_t^2),
$$

$$
o_t^2=(p_t^2,p_t^1),
$$

and learns an independent Q-function:

$$
Q_1(o_1,a_1),
\qquad
Q_2(o_2,a_2).
$$

The notebook explains why multi-agent learning is harder due to non-stationarity, coordination, state/action-space growth, credit assignment, partial observability, and interacting exploration.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The notebooks are designed to run in **Google Colab** or a standard Jupyter environment.

For the Gymnasium examples, the requirements include the Toy Text, Classic Control, and Atari dependencies.

### Atari / MarioBros note

The MarioBros notebook uses:

```text
ALE/MarioBros-v5
```

Atari environments require ALE support and the appropriate game ROM. If your local environment reports that the ROM is unavailable, install the Atari ROMs supported by your Gymnasium/ALE setup before running that notebook.

## Running the Portfolio

Open the notebooks in numerical order.

For Jupyter:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

For Google Colab, upload an individual `.ipynb` file and run the cells from top to bottom.

Some notebooks generate output files such as:

- learned Q-table CSV files;
- simulation GIFs;
- policy and convergence plots.

## Main Python Libraries

- NumPy
- Pandas
- Matplotlib
- Gymnasium
- ImageIO
- ALE-Py
- IPython / Jupyter

## Portfolio Progression

The folder follows this learning progression:

$$
\text{TD Control Foundations}
\rightarrow
\text{Q-Learning / SARSA / Expected SARSA}
\rightarrow
\text{Robot Navigation}
\rightarrow
\text{Multi-Agent Q-Learning}
\rightarrow
\text{Gymnasium Control Problems}
\rightarrow
\text{Continuous-State Discretization}
\rightarrow
\text{High-Dimensional State Aggregation}
$$

The later notebooks also show why tabular methods eventually become difficult as state and action spaces grow, motivating function approximation and deep reinforcement learning.
