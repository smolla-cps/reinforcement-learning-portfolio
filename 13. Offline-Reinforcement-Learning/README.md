# Offline Reinforcement Learning

A portfolio collection of **offline reinforcement learning** projects that learn policies from fixed datasets rather than collecting new training experience through online environment interaction.

The projects progress from a small tabular problem to discrete offline deep reinforcement learning and then to continuous, goal-conditioned control.

## Learning Progression

**Fixed Dataset → Offline Q-Learning → Conservative Offline RL → Batch-Constrained RL → Implicit Q-Learning → Continuous Offline RL**

| # | Notebook | Environment / Dataset | Methods | Action Space | Main Focus |
|---|---|---|---|---|---|
| 01 | `01_Offline_RL_GridWorld_10x10.ipynb` | Custom 10×10 GridWorld + fixed CSV dataset | Offline Q-Learning from scratch | Discrete | Offline RL fundamentals, dataset inspection, supported actions, tabular learning, and Gymnasium evaluation |
| 02 | `02_Offline_RL_FourRooms_CQL_BCQ.ipynb` | Minari `D4RL/minigrid/fourrooms-random-v0` | CQL from scratch, d3rlpy Discrete CQL, Discrete BCQ | Discrete | Conservative value learning, batch constraints, Minari datasets, and library-based offline RL |
| 03 | `03_Offline_RL_D4RL_PointMaze_IQL_CQL.ipynb` | Minari `D4RL/pointmaze/medium-v2` | IQL from scratch, d3rlpy IQL, Continuous CQL | Continuous | Goal-conditioned continuous control, PyTorch implementation, offline actor-critic learning, and D4RL/Minari workflows |

## What Is Offline Reinforcement Learning?

In online reinforcement learning, the agent repeatedly interacts with an environment:

```math
S_t \rightarrow A_t \rightarrow R_{t+1},S_{t+1}
```

and uses newly collected experience to improve its policy.

In **offline reinforcement learning**, the training data already exists. The learner receives a fixed dataset such as:

```math
\mathcal D
=
\left\{
(S_t,A_t,R_{t+1},S_{t+1},d_t)
\right\}
```

and must learn without collecting additional training transitions.

This creates an important challenge: the learned policy may prefer actions that are poorly represented or completely absent from the dataset.

The notebooks in this folder examine increasingly sophisticated ways to handle that problem.

## 01 — Offline Q-Learning on a 10×10 GridWorld

The first project introduces offline RL with a controlled discrete environment and a fixed CSV transition dataset.

### Main topics

- Complete inspection of a fixed offline dataset
- State, action, reward, next-state, and done fields
- Episode-level dataset statistics
- State-action coverage
- Offline action support
- Tabular offline Q-learning from scratch
- Greedy supported policy extraction
- Plain-Python environment dynamics
- Equivalent Gymnasium environment
- Random-policy baseline
- Policy evaluation

The central learning rule is the Q-learning target:

```math
Y_t
=
R_{t+1}
+
\gamma
\max_{a'}
Q(S_{t+1},a')
```

but the maximization is restricted to actions supported by the offline dataset.

This project establishes the core distinction between **learning from a fixed dataset** and online interaction.

### Dataset note

The notebook is designed to upload the GridWorld CSV interactively in Google Colab. A local dataset path can also be supplied by changing `DATASET_FILE`.

## 02 — Offline RL in FourRooms: CQL and BCQ

The second project moves from tabular offline learning to a larger discrete offline-RL problem using the Minari dataset:

```text
D4RL/minigrid/fourrooms-random-v0
```

The notebook first inspects the dataset in detail, including:

- observation structure
- action space
- complete episodes
- transition alignment
- termination and truncation
- action distribution
- reward distribution
- episode returns
- episode lengths
- dataset success rate

It then compares multiple offline-RL approaches.

### Conservative Q-Learning from scratch

CQL adds a conservative penalty that discourages unrealistically large Q-values for actions that are not well supported by the dataset.

The idea is to reduce the tendency of ordinary Q-learning to overestimate out-of-distribution actions.

### Discrete CQL with d3rlpy

The same problem is then implemented with the `d3rlpy` offline reinforcement learning library.

This demonstrates the transition from a manually implemented algorithm to a reusable offline-RL framework.

### Discrete BCQ

Batch-Constrained Q-Learning restricts policy decisions toward actions that resemble those contained in the offline dataset.

The notebook therefore compares two important offline-RL ideas:

```math
\boxed{\text{CQL: conservative value estimation}}
```

and

```math
\boxed{\text{BCQ: constrain actions toward dataset support}}.
```

## 03 — D4RL PointMaze: IQL and Continuous CQL

The third project addresses continuous, goal-conditioned offline reinforcement learning using:

```text
D4RL/pointmaze/medium-v2
```

through Minari and Gymnasium-Robotics.

The observation is goal-conditioned and the action space is continuous.

The notebook performs detailed dataset inspection before training, including:

- observation-space structure
- continuous action components
- Minari episode structure
- aligned transition arrays
- termination and truncation handling
- episode statistics
- reward statistics
- state-feature statistics
- action distributions

### Implicit Q-Learning from scratch

A manual PyTorch implementation introduces the main IQL components:

- Q-networks
- state-value network
- expectile regression
- target-network updates
- advantage-weighted policy learning
- random offline mini-batches

IQL avoids directly maximizing Q-values over unseen actions and instead learns from actions already present in the dataset.

### IQL with d3rlpy

The notebook then trains a library-based IQL implementation for comparison with the manual implementation.

### Continuous CQL

Continuous CQL provides an additional conservative offline-RL method for the same PointMaze dataset.

This creates a useful comparison between:

```math
\boxed{\text{IQL}}
```

and

```math
\boxed{\text{CQL}}
```

for continuous offline control.

## Portfolio Progression

The three projects are intentionally ordered by increasing complexity.

### Stage 1 — Small discrete offline RL

```text
10×10 GridWorld
        ↓
Fixed CSV dataset
        ↓
Offline Q-Learning
```

### Stage 2 — Conservative discrete offline RL

```text
FourRooms
        ↓
Minari dataset
        ↓
CQL from scratch
        ↓
d3rlpy CQL
        ↓
Discrete BCQ
```

### Stage 3 — Continuous offline RL

```text
D4RL PointMaze
        ↓
Goal-conditioned continuous state/action problem
        ↓
Manual IQL with PyTorch
        ↓
d3rlpy IQL
        ↓
Continuous CQL
```

## Libraries and Their Roles

### Gymnasium

Provides standardized reinforcement learning environments and evaluation interfaces.

### Minari

Provides standardized offline reinforcement learning datasets and episode access.

### MiniGrid

Provides the FourRooms-style environment required for the discrete Minari project.

### Gymnasium-Robotics

Provides the modern Gymnasium robotics environments used by the PointMaze project.

### MuJoCo

Provides the physics engine required by the PointMaze environment.

### d3rlpy

Provides implementations of offline deep reinforcement learning algorithms such as:

- Discrete CQL
- Discrete BCQ
- IQL
- Continuous CQL

### PyTorch

Used for the from-scratch IQL implementation in the PointMaze notebook.

## Core Offline-RL Concepts Demonstrated

Across the three notebooks, the portfolio covers:

- Fixed offline datasets
- Transition tuples
- Dataset coverage
- Behavior-policy data
- Supported and unsupported actions
- Distribution shift
- Out-of-distribution action risk
- Offline Q-learning
- Conservative Q-Learning
- Batch-Constrained Q-Learning
- Implicit Q-Learning
- Expectile regression
- Advantage-weighted policy learning
- Replay-style offline mini-batch sampling
- Target networks
- Soft target updates
- Discrete and continuous action spaces
- Goal-conditioned observations
- Dataset inspection and validation
- Termination vs. truncation
- Random-policy baselines
- Policy evaluation
- Average return
- Success rate
- Final goal distance
- Algorithm comparison

## Folder Structure

```text
offline-reinforcement-learning/
│
├── 01_Offline_RL_GridWorld_10x10.ipynb
├── 02_Offline_RL_FourRooms_CQL_BCQ.ipynb
├── 03_Offline_RL_D4RL_PointMaze_IQL_CQL.ipynb
├── README.md
└── requirements.txt
```

## Installation

Install the shared dependencies with:

```bash
pip install -r requirements.txt
```

Then start JupyterLab:

```bash
jupyter lab
```

The Minari notebooks download their referenced datasets when they are first run, so an internet connection is required for the initial dataset download.

The GridWorld notebook instead prompts for its fixed CSV dataset when run in Google Colab.

## Recommended Order

Run the notebooks in numerical order.

1. Start with the 10×10 GridWorld to understand the basic offline-RL data structure and offline Q-learning.
2. Continue to FourRooms to study conservative and batch-constrained learning with a standardized offline dataset.
3. Finish with D4RL PointMaze to study deep continuous offline reinforcement learning with IQL and CQL.

## Main Technologies

- Python
- Gymnasium
- Minari
- MiniGrid
- Gymnasium-Robotics
- MuJoCo
- d3rlpy
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Jupyter

## Portfolio Takeaway

This folder demonstrates a progression from manually implemented offline value learning to modern deep offline reinforcement learning.

The projects emphasize not only algorithm implementation, but also the central challenge of offline RL:

```math
\boxed{
\text{learn a strong policy without interacting with the environment during training}
}
```

while remaining aware of the limitations imposed by the fixed dataset and its action/state coverage.
