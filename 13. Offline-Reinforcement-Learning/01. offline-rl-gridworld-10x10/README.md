# Offline Reinforcement Learning on a 10 × 10 GridWorld

## Overview

This portfolio project builds a complete offline reinforcement-learning problem from the ground up.

The workflow is:

```text
Define a 10 × 10 GridWorld
        ↓
Generate a fixed offline dataset
        ↓
Inspect states, actions, rewards, episodes, and transitions
        ↓
Train support-constrained offline Q-learning from scratch
        ↓
Evaluate the learned policy
        ↓
Represent the same problem with Gymnasium
        ↓
Train again from the exact same fixed dataset
        ↓
Compare with a random baseline
```

The main training dataset is:

```text
data/gridworld_offline_dataset.csv
```

CSV is used intentionally because it is easy to inspect, upload to Google Colab, and read with plain Python without Excel-reading packages.

---

# What Is the Environment?

The environment is a two-dimensional `10 × 10` grid.

- Start: `(0,0)`
- Goal: `(9,9)`
- Blocked cells: 19
- Actions: Up, Right, Down, Left

```text
    0 1 2 3 4 5 6 7 8 9
0   S . . . . . . . . .
1   . . # . . . . . . .
2   . . # . . . # . # .
3   . . # . . . # . # .
4   . . # # # # # . # .
5   . . . . . . # . # .
6   . . . . . . # . . .
7   . . . . . . # # # .
8   . . . . . . . . . .
9   . . . . . . . . . G
```

Legend:

```text
S = start
G = goal
# = blocked
. = normal traversable cell
```

For a complete transition rule, trying to leave the grid is treated the same as hitting a blocked cell:

```text
reward = -5
next state = current state
```

---

# What Is the Dataset About?

The dataset contains previously collected trajectories through this GridWorld.

Every row represents one fixed offline transition:

```text
current state
    ↓
logged action
    ↓
reward
    ↓
next state
    ↓
episode-ending information
```

The minimum RL representation is:

```text
(state, action, reward, next_state, done)
```

The dataset was collected before offline training begins.

The learner is not allowed to collect additional training transitions.

---

# What Is the Offline RL Agent Trying to Learn?

The agent wants to learn:

```text
policy(state) → action
```

More specifically:

> Given the current `(row, column)`, should the agent move Up, Right, Down, or Left so that it reaches `(9,9)` while avoiding blocked cells and boundary collisions?

The reward structure teaches the following preference:

```text
normal valid move       →  0
blocked/boundary hit    → -5
reach goal              → +10
```

Repeated Bellman updates propagate the positive goal value backward through earlier states.

---

# State Space

The state is:

```text
(row, column)
```

where both coordinates range from `0` to `9`.

For tabular learning, each position is encoded as:

```text
State_ID = row * 10 + column
```

Examples:

```text
(0,0) → 0
(0,9) → 9
(1,0) → 10
(9,9) → 99
```

The encoded state space therefore contains 100 possible IDs.

Blocked cells are included in the map but cannot be entered.

---

# Action Space

| Action ID | Action | Row change | Column change |
|---:|---|---:|---:|
| 0 | Up | -1 | 0 |
| 1 | Right | 0 | +1 |
| 2 | Down | +1 | 0 |
| 3 | Left | 0 | -1 |

There are four discrete actions.

---

# Reward Function

The project uses exactly the requested reward structure:

```text
Goal reached             +10
Blocked cell hit          -5
Normal valid position      0
```

For completeness, a boundary collision is also assigned `-5`.

A blocked or boundary collision leaves the agent in the same state.

---

# Episode Ending

An episode can end in two ways.

## Terminated

The goal `(9,9)` is reached.

## Truncated

The agent has not reached the goal after 100 steps.

The dataset also stores:

```text
Done = Terminated OR Truncated
```

---

# How Was the Offline Dataset Generated?

A useful offline dataset should not contain only perfect demonstrations.

The generated data deliberately mix successful behavior with mistakes.

## Mixed-guided episodes

About 80% of episodes use a noisy goal-directed behavior policy.

At each step:

```text
65% → choose an action that reduces obstacle-aware distance to the goal
35% → choose a random action
```

## Pure-random episodes

About 20% of episodes use random actions only.

With random seed 42, the actual generated dataset contains:

| Property | Value |
|---|---:|
| Episodes | 1,000 |
| Transitions | 42,986 |
| Mixed-guided episodes | 793 |
| Pure-random episodes | 207 |
| Goal-reaching episodes | 797 |
| Truncated episodes | 203 |
| Behavior-data success rate | 79.7% |
| Average episode length | 42.99 |
| Average episode return | -28.61 |
| Blocked-cell hits | 2,842 |
| Boundary hits | 4,473 |

The behavior policy is used only to create the fixed data. It is not used by the offline learner.

---

# Dataset Files

```text
data/
├── gridworld_offline_dataset.csv   # complete training transitions
├── episode_summary.csv             # one row per episode
├── grid_layout.csv                 # visual grid representation
├── blocked_cells.csv               # blocked coordinates
├── actions.csv                     # action definitions
└── README.md
```

The full transition CSV contains:

- episode number
- step number
- behavior type
- current row/column
- current state ID
- action ID/name
- immediate reward
- next row/column
- next state ID
- terminated
- truncated
- done
- blocked/boundary indicators
- obstacle-aware distance-to-goal information

The offline learner fundamentally needs only:

```text
State_ID
Action_ID
Reward
Next_State_ID
Done
```

---

# Method 1 — Offline Q-Learning from Scratch

The first training implementation deliberately uses:

```text
NO import statements
NO NumPy
NO Gymnasium
NO PyTorch
NO reinforcement-learning package
```

The CSV is read using only built-in:

```python
open(...)
split(...)
```

and the Q-table is a Python list of lists.

## Why Support-Constrained Offline Q-Learning?

Offline learning cannot safely assume that every possible action is represented in the fixed data.

For each state, the notebook first records which actions were actually observed.

During the Bellman backup, the next-state maximum is taken only over observed actions.

Conceptually:

```text
Q(s,a) ← Q(s,a)
       + learning_rate
       × [reward
          + gamma × best_supported_next_Q
          - Q(s,a)]
```

This is a simple offline adaptation that avoids relying on totally unsupported actions.

## Pros

- transparent
- easy to explain
- no RL library
- ideal for a small discrete problem
- shows exactly how fixed transitions update Q-values

## Cons

- tabular only
- cannot generalize between similar states
- depends on dataset coverage
- not appropriate for high-dimensional or continuous problems

---

# Method 2 — Same Problem with Gymnasium

The second implementation defines the same problem as:

```python
gym.Env
```

with:

```text
observation_space = Discrete(100)
action_space      = Discrete(4)
```

The same fixed CSV is used for training.

This distinction is important:

> Gymnasium standardizes the environment. It does not automatically make the learning online or offline.

During the second training section:

```text
env.step() is NOT used to collect training data
```

The environment is used only to:

- formalize state/action spaces
- verify transition behavior
- evaluate the learned policy

The Q-table is implemented with NumPy for convenience, while the learning rule remains the same.

---

# Dataset Verification

Before packaging the project, the fixed dataset was tested with the from-scratch support-constrained offline Q-learning method.

The learned greedy policy successfully reaches the goal in 18 steps.

One learned path is:

```text
(0,0)
(1,0)
(2,0)
(2,1)
(3,1)
(4,1)
(5,1)
(6,1)
(6,2)
(7,2)
(8,2)
(8,3)
(8,4)
(8,5)
(8,6)
(8,7)
(8,8)
(9,8)
(9,9)
```

This confirms that the dataset contains enough useful information for the simple offline learner.

---

# Repository Structure

```text
offline-rl-gridworld-10x10/
│
├── README.md
├── requirements.txt
├── .gitignore
├── generate_dataset.py
│
├── data/
│   ├── README.md
│   ├── gridworld_offline_dataset.csv
│   ├── episode_summary.csv
│   ├── grid_layout.csv
│   ├── blocked_cells.csv
│   └── actions.csv
│
└── notebooks/
    └── offline_rl_gridworld_10x10.ipynb
```

---

# Installation

The first half of the notebook needs no third-party packages.

For the Gymnasium section:

```bash
pip install -r requirements.txt
```

The requirements are intentionally minimal:

```text
gymnasium
numpy
matplotlib
jupyter
```

---

# Google Colab

Upload:

```text
gridworld_offline_dataset.csv
```

and change:

```python
DATASET_FILE = "../data/gridworld_offline_dataset.csv"
```

to:

```python
DATASET_FILE = "/content/gridworld_offline_dataset.csv"
```

---

# Evaluation

The notebook reports:

- whether the learned policy reaches the goal
- total reward
- number of steps
- complete path
- learned policy symbols across the grid
- random-policy baseline performance

The from-scratch and Gymnasium versions intentionally use the same training data and same Bellman logic.

---

# Important Reward Limitation

A normal move receives reward `0`.

Therefore a safe 18-step path and a safe 25-step path can ultimately receive the same goal reward.

The current objective encourages:

```text
reach the goal
+
avoid -5 collisions
```

It does not explicitly encourage the shortest possible path.

If shortest-path efficiency becomes important, a future version could add a small normal-step cost such as `-0.1`.

---

# Portfolio Learning Outcomes

This project demonstrates:

- Markov decision process design
- offline dataset generation
- behavior-policy data collection
- transition data structure
- state/action/reward/next-state/done representation
- episode termination and truncation
- offline Q-learning
- Bellman updates
- support-constrained backups
- Q-table policy extraction
- Gymnasium environment creation
- fixed-dataset training
- random baseline evaluation
- offline vs online RL distinction

---

# Final Takeaway

The key idea is simple:

> The policy learns entirely from a fixed CSV of previously collected transitions.

The first implementation exposes the mechanics with plain Python.

The second implementation shows how the same problem fits the standard Gymnasium environment API without changing the offline nature of training.
