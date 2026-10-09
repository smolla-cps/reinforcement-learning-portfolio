# Offline Reinforcement Learning: From Raw Minari Data to CQL and BCQ

## Overview

This portfolio project studies **offline reinforcement learning (Offline RL)** from the data level upward.

The project deliberately follows a transparent progression:

1. Download and inspect a real public offline RL dataset from **Minari**.
2. Print the complete dataset structure, spaces, shapes, data types, episodes, rewards, termination flags, truncation flags, and example transitions.
3. Implement **Conservative Q-Learning (CQL)** from scratch using explicit Python logic rather than an RL algorithm library.
4. Implement **Discrete CQL** again using the **d3rlpy** offline RL library.
5. Implement a second offline RL method, **Discrete Batch-Constrained Q-Learning (Discrete BCQ)**, using d3rlpy.
6. Evaluate all learned policies in the recovered Gymnasium environment.
7. Compare them against a random baseline.

The notebook is intentionally educational. It keeps the algorithmic steps visible instead of hiding the entire workflow behind high-level library calls.

---

## Why Offline RL?

Online reinforcement learning learns by repeatedly interacting with an environment:

```text
State → Action → Environment → Reward → Update → Repeat
```

Offline reinforcement learning learns from an already collected dataset:

```text
Fixed Dataset
     ↓
(s, a, r, s', termination/truncation)
     ↓
Offline RL Algorithm
     ↓
Learned Policy
```

During offline training, the policy **does not collect new training experience**.

This makes offline RL useful when online exploration is expensive, slow, unsafe, or impossible.

---


## What Is This Dataset About?

This project uses the Minari dataset:

```text
D4RL/minigrid/fourrooms-random-v0
```

The dataset contains previously collected trajectories from the **MiniGrid FourRooms** navigation environment.

In FourRooms, an agent is placed in a grid world divided into four connected rooms. The agent must navigate through doorways and reach a goal location.

Each recorded transition contains information about:

```text
current observation
        ↓
action taken
        ↓
reward received
        ↓
next observation
        ↓
episode termination information
```

The dataset was collected using a **random behavior policy**. That means the recorded agent was not an expert. It selected actions randomly while interacting with the environment.

So the dataset represents:

> **Previously recorded navigation experience from a randomly acting agent in the FourRooms environment.**

---

## What Is the Offline RL Agent Trying to Learn?

The offline RL algorithm does not control the environment while training.

Instead, it studies the fixed dataset and tries to learn a policy:

\[
\pi(s) \rightarrow a
\]

that answers:

> **Given the current FourRooms state, which action should the agent choose so that it is more likely to reach the goal and obtain a high long-term return?**

For this task:

```text
State
  ↓
7 × 7 × 3 encoded local grid
+
agent direction

Policy
  ↓

Action
  ↓
turn left
turn right
move forward
...
```

The desired learned behavior is:

```text
observe the current grid
        ↓
understand where the agent is facing
        ↓
choose useful navigation actions
        ↓
reach the goal efficiently
```

The learning objective is therefore to maximize expected cumulative reward:

\[
\max_\pi
\mathbb{E}
\left[
\sum_t \gamma^t r_t
\right]
\]

The important offline-RL constraint is that the algorithm must learn this policy **only from the recorded dataset**. It cannot collect new training experience by interacting with FourRooms.

---

## Dataset vs Learning Problem

| Question | Answer |
|---|---|
| **What is the environment?** | MiniGrid FourRooms navigation |
| **What is stored in the dataset?** | Previously collected state-action-reward trajectories |
| **Who generated the data?** | A random behavior policy |
| **What does the state describe?** | Local grid structure and agent direction |
| **What is the action?** | One of seven discrete MiniGrid actions |
| **What does reward represent?** | Progress/success in reaching the goal |
| **What does offline RL learn?** | A policy that chooses better actions from each state |
| **What is the final objective?** | Reach the goal with high cumulative reward |


## Dataset

The project uses the Minari dataset:

```text
D4RL/minigrid/fourrooms-random-v0
```

The dataset was generated in the **MiniGrid FourRooms** environment by sampling random actions.

According to the current Minari dataset page, it contains:

- **1,000,070 transitions**
- **10,174 episodes**
- observation space containing:
  - `direction: Discrete(4)`
  - `image: Box(0, 255, (7, 7, 3), uint8)`
  - `mission: Text(...)`
- action space:
  - `Discrete(7)`

Official dataset page:
https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-random-v0/

---

## What the Notebook Inspects

Before training any model, the notebook prints and analyzes:

- installed library versions
- dataset object type
- dataset ID
- total episodes
- total transitions
- complete observation space
- every observation-space component
- image shape and data type
- image bounds
- direction space
- mission space
- complete action space
- number of actions
- human-readable action mapping
- first episode object type
- episode ID
- episode observation keys
- observation array shapes
- observation data types
- action shape and type
- reward shape and type
- termination shape and type
- truncation shape and type
- optional `infos`
- relationship between `T` actions and `T+1` observations
- one complete transition
- current state
- action
- reward
- next state
- `terminated`
- `truncated`
- `done`
- episode return
- episode length
- action distribution
- reward statistics
- positive-reward frequency
- episode-return distribution
- episode-length distribution
- success frequency
- termination/truncation counts
- flattened state size used by deep RL methods

This makes the dataset section understandable before any reinforcement learning code appears.

---

# Method 1: Conservative Q-Learning (CQL)

## Why CQL?

A major problem in offline RL is **distribution shift**.

The dataset only tells us what happened for actions that the behavior policy actually took. A learned Q-function can nevertheless assign unrealistically high values to other actions. If the policy greedily chooses those overestimated actions, performance can collapse.

CQL addresses this by learning a **conservative Q-function**.

For discrete actions, its objective contains a Bellman-learning term plus a conservative penalty of the form

\[
\alpha
\left[
\log \sum_a \exp Q(s,a)
-
Q(s,a_D)
\right]
\]

where \(a_D\) is the action recorded in the dataset.

The penalty pushes down Q-values broadly while preserving or increasing support for actions represented in the dataset.

### Why CQL is the main method in this project

CQL is a strong choice here because:

- the task has a **discrete action space**
- the data are fixed
- the behavior policy is random
- offline Q-value overestimation is directly relevant
- the conservative penalty is easy to motivate mathematically
- d3rlpy provides a dedicated `DiscreteCQL` implementation

### Pros

- specifically designed for offline RL
- explicitly addresses Q-value overestimation
- does not require online exploration during training
- works naturally with discrete control
- strong educational connection to standard Q-learning

### Cons

- conservative strength must be tuned
- too much conservatism can underestimate useful actions
- deep CQL introduces neural-network and optimization complexity
- performance still depends heavily on dataset coverage and quality
- sparse successful trajectories can make learning difficult

---

## CQL Implementation A: From Scratch

The notebook first implements a **tabular/discrete CQL objective from scratch**.

No RL algorithm library is used in this section.

The code explicitly performs:

- Q-table creation
- Bellman target computation
- temporal-difference error computation
- stabilized softmax computation
- conservative-gradient computation
- manual Q-value updates
- policy extraction

This version is intentionally simple and transparent. It is an educational tabular adaptation of the discrete CQL objective rather than a neural reproduction of the original paper.

---

## CQL Implementation B: d3rlpy

The same concept is then implemented with:

```python
d3rlpy.algos.DiscreteCQLConfig
```

The library implementation is a deep offline RL method based on Double DQN with the CQL conservative term.

Gymnasium is used for environment evaluation, Minari supplies the offline dataset, and d3rlpy supplies the offline RL algorithm.

---

# Method 2: Discrete Batch-Constrained Q-Learning (BCQ)

The second offline RL method is **Discrete BCQ**.

BCQ uses a behavior/imitation model to estimate which actions are sufficiently likely under the dataset. The Q-policy is then restricted to those supported actions.

Conceptually:

```text
Offline Dataset
      ↓
Learn behavior-action probabilities
      ↓
Filter unlikely actions
      ↓
Choose highest-Q action among supported actions
```

### Why compare BCQ with CQL?

They address the same offline-RL problem differently:

- **CQL:** penalizes overly optimistic Q-values.
- **BCQ:** constrains which actions the learned policy is allowed to consider.

### Pros

- explicitly limits out-of-distribution actions
- intuitive for discrete action spaces
- combines behavior modeling and value learning
- useful comparison against CQL

### Cons

- sensitive to the action-filter threshold
- depends on how well the behavior model represents the dataset
- if the behavior data are poor, staying close to them can limit performance
- on a highly random dataset, many actions may appear plausible, weakening the filtering effect

---

## Why Not IQL Here?

Implicit Q-Learning (IQL) is also an important modern offline RL algorithm, but this repository starts with **two methods that have explicit discrete-control implementations in d3rlpy** for this MiniGrid task:

- `DiscreteCQL`
- `DiscreteBCQ`

This keeps the comparison technically aligned with the dataset's `Discrete(7)` action space.

---

## Repository Structure

```text
offline-rl-fourrooms/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── offline_rl_fourrooms.ipynb
│
└── data/
    └── README.md
```

---

## Notebook Structure

### Part I — Dataset Understanding

1. Project objective
2. Offline RL vs Online RL
3. Install dependencies
4. Import packages and print versions
5. Load the Minari dataset
6. Inspect dataset object and metadata
7. Inspect complete observation space
8. Inspect every observation feature
9. Inspect complete action space
10. Print human-readable action mapping
11. Inspect one full episode
12. Verify `observations = actions + 1`
13. Inspect state feature values
14. Print one complete transition
15. Explain termination vs truncation vs done
16. Inspect global action distribution
17. Inspect reward distribution
18. Inspect episode returns
19. Inspect episode lengths
20. Inspect success/termination/truncation frequencies
21. Build a dataset summary table
22. Define the 148-feature vector used by deep algorithms

### Part II — CQL from Scratch

23. Explain CQL
24. CQL pros and cons
25. Define hyperparameters
26. Build Q-table
27. Manual conservative update
28. Train entirely from fixed data
29. Extract policy
30. Evaluate in Gymnasium

### Part III — CQL with d3rlpy

31. Explain the role of Minari, Gymnasium, and d3rlpy
32. Convert Minari Dict observations to numeric vectors
33. Build a d3rlpy offline dataset
34. Train `DiscreteCQL`
35. Evaluate CQL

### Part IV — Discrete BCQ with d3rlpy

36. Explain BCQ
37. BCQ pros and cons
38. Train `DiscreteBCQ`
39. Evaluate BCQ

### Part V — Comparison

40. Random baseline
41. Compare all methods
42. Plot average return
43. Plot success rate
44. Discuss results and limitations
45. Future improvements

---

## Installation

A clean environment is recommended.

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

---

## Libraries and Their Roles

| Library | Role |
|---|---|
| Minari | Downloads and reads the fixed offline dataset |
| Gymnasium | Provides/reconstructs the environment for evaluation |
| MiniGrid | Supplies the FourRooms environment |
| d3rlpy | Implements deep offline RL algorithms |
| NumPy | Numeric dataset conversion |
| Pandas | Inspection and result tables |
| Matplotlib | Visualization |

A key distinction is:

> **Gymnasium does not implement CQL or BCQ.**

Gymnasium provides the environment interface.  
The offline RL algorithms are implemented by **d3rlpy**.

---

## Evaluation

All methods are evaluated with the same set of Gymnasium seeds.

Primary metrics:

### Average Return

\[
\text{Average Return}
=
\frac{1}{N}
\sum_{i=1}^{N} R_i
\]

### Success Rate

\[
\text{Success Rate}
=
\frac{\text{Successful Episodes}}
{\text{Evaluation Episodes}}
\times 100
\]

The final comparison contains:

| Policy | Average Return | Success Rate |
|---|---:|---:|
| Random Policy | measured | measured |
| CQL from Scratch | measured | measured |
| Discrete CQL — d3rlpy | measured | measured |
| Discrete BCQ — d3rlpy | measured | measured |

No result values are hard-coded; they must come from running the experiment.

---

## Important Methodological Note

The **from-scratch CQL section is educational and tabular**. It exposes the conservative update directly.

The d3rlpy `DiscreteCQL` implementation is the more realistic deep-RL implementation for comparison and portfolio demonstration.

This distinction is stated explicitly so the project does not imply that the hand-written tabular implementation is a complete reproduction of the original neural CQL paper.

---

## Future Extensions

- compare random and expert Minari datasets
- Behavior Cloning baseline
- Fitted Q Evaluation
- hyperparameter sensitivity for CQL `alpha`
- hyperparameter sensitivity for BCQ action flexibility
- neural CQL implemented directly in PyTorch
- repeated-seed statistical analysis
- additional discrete offline RL benchmark datasets

---

## References

- Minari FourRooms dataset: https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-random-v0/
- Minari: https://minari.farama.org/
- Gymnasium: https://gymnasium.farama.org/
- d3rlpy: https://d3rlpy.readthedocs.io/
- CQL: Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*
- BCQ: Fujimoto et al., *Off-Policy Deep Reinforcement Learning without Exploration*
