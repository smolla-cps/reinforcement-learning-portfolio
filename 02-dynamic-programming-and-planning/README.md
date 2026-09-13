# Dynamic Programming for MDPs and Model-Based Planning

> **Math rendering:** Inline mathematical notation uses `$...$`. Display equations use GitHub's fenced `math` blocks for reliable rendering in README files.

This section demonstrates two major approaches to planning when the environment dynamics are known or can be simulated:

1. **Dynamic Programming for Markov Decision Processes**
   - Policy Evaluation
   - Policy Iteration
   - Value Iteration

2. **Search-Based Model Planning**
   - Monte Carlo Tree Search

The notebooks progress from small, transparent numerical examples to complete stochastic planning problems.

```text
Known or simulatable environment model
                ↓
             Planning
                │
        ┌───────┴────────┐
        ↓                ↓
Dynamic Programming     Tree Search
        ↓                ↓
Policy Evaluation       MCTS
Policy Iteration
Value Iteration
```

Dynamic Programming repeatedly performs Bellman backups over the state space. Monte Carlo Tree Search instead builds a selective search tree around the current decision and estimates promising actions through simulated futures.

---

## Project Structure

```text
02-dynamic-programming-and-planning/
│
├── README.md
├── requirements.txt
├── 01. 3x4grid_value_policy_iteration_step_by_step.ipynb
├── 02. frozenlake_dynamic_programming.ipynb
└── 03. tic_tac_toe_monte_carlo_tree_search.ipynb
```

---

# 1. 3×4 Gridworld: Value Iteration and Policy Iteration Step by Step

The first notebook uses a small stochastic $3\times4$ Gridworld so that every Dynamic Programming calculation can be inspected directly.

The environment uses:

- intended movement probability $0.8$,
- perpendicular slip probability $0.1$ on each side,
- one wall,
- terminal values $+1$ and $-1$,
- discount factor $\gamma=0.9$.

The notebook demonstrates:

- the MDP structure,
- stochastic transitions,
- the Bellman optimality equation,
- synchronous Value Iteration,
- iteration-by-iteration value propagation,
- manual calculations for individual states,
- Policy Evaluation,
- Policy Improvement,
- complete Policy Iteration,
- final value functions,
- final greedy policies,
- and direct comparison between Policy Iteration and Value Iteration.

The executed notebook converges to:

```text
 0.64   0.74   0.85   1.00
 0.57   WALL   0.57  -1.00
 0.49   0.43   0.48   0.28
```

with the greedy policy:

```text
→   →   →   +1
↑  WALL ↑   -1
↑   ←   ↑    ←
```

Value Iteration converges after $35$ sweeps. Policy Iteration reaches the same final value function and the same action in all $9$ non-terminal states.

---

# 2. Slippery FrozenLake: Dynamic Programming on an MDP

The second notebook applies Dynamic Programming to the standard $4\times4$ slippery FrozenLake benchmark.

```text
S F F F
F H F H
F F F H
H F F G
```

The model is implemented directly so that the transition probabilities and rewards can be inspected rather than hidden behind an external environment package.

## MDP Components

### State

There are $16$ states numbered from $0$ to $15$.

### Action

The action space is:

```text
0 = Left
1 = Down
2 = Right
3 = Up
```

### Transition

The environment is slippery. For a selected action, one of three movement directions occurs with probability $1/3$.

The transition model is therefore:

```math
P(s'|s,a).
```

### Reward

The notebook uses a transition-reward convention:

```math
R(s,a,s')=1
```

when the transition enters the goal and $0$ otherwise.

## Topics Demonstrated

The notebook includes:

- exact transition-model construction,
- model verification,
- action-value calculation directly from $P(s'|s,a)$ and $R(s,a,s')$,
- Policy Evaluation of a uniform random policy,
- Bellman expectation verification,
- Policy Iteration,
- Value Iteration,
- convergence histories,
- greedy-policy extraction,
- model-backup counts,
- algorithm comparison,
- and simulation-based evaluation of the resulting policies.

With $\gamma=0.99$ and $\theta=10^{-10}$:

- random-policy evaluation converges in $93$ sweeps,
- Policy Iteration converges in $7$ outer iterations,
- Value Iteration converges in $571$ Bellman optimality sweeps,
- the final start-state value is approximately $0.5420$,
- the maximum difference between the final Policy Iteration and Value Iteration value functions is approximately $1.3\times10^{-11}$,
- the final policies have complete action agreement,
- and both optimal policies obtain an empirical success rate of approximately $0.7487$ over $3000$ stochastic simulation episodes under the notebook settings.

---

# Dynamic Programming Equations

## Policy Evaluation

Policy Evaluation asks:

> What is the expected return from each state if the current policy continues to be followed?

For a stochastic policy $\pi(a|s)$:

```math
V_{k+1}^{\pi}(s)
=
\sum_a
\pi(a|s)
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V_k^{\pi}(s')
\right].
```

For a deterministic policy:

```math
V_{k+1}^{\pi}(s)
=
\sum_{s'}
P(s'|s,\pi(s))
\left[
R(s,\pi(s),s')
+
\gamma V_k^{\pi}(s')
\right].
```

Policy Evaluation does **not** take the maximum over actions. It evaluates the action or action distribution defined by the current policy.

---

## Policy Improvement

After $V^\pi$ has converged, action values are calculated as:

```math
Q^\pi(s,a)
=
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^\pi(s')
\right].
```

The improved policy is:

```math
\pi_{\text{new}}(s)
=
\arg\max_a Q^\pi(s,a).
```

Policy Iteration repeatedly performs:

```text
Current policy
      ↓
Policy Evaluation
      ↓
V^π converges
      ↓
Policy Improvement
      ↓
argmax over actions
      ↓
New policy
      ↓
Repeat until the policy is unchanged
```

---

## Value Iteration

Value Iteration directly applies the Bellman optimality update:

```math
V_{k+1}(s)
=
\max_a
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V_k(s')
\right].
```

For every state:

```text
Calculate Q(s, Left)
Calculate Q(s, Down)
Calculate Q(s, Right)
Calculate Q(s, Up)
        ↓
Take max
        ↓
Store V(k+1)(s)
```

The $\max$ gives the best numerical value.

After convergence, the greedy optimal policy is:

```math
\pi^*(s)
=
\arg\max_a
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^*(s')
\right].
```

The $\arg\max$ gives the action that produces the largest expected value.

---

# 3. Monte Carlo Tree Search for Tic-Tac-Toe

The third notebook demonstrates Monte Carlo Tree Search as a model-based, search-oriented planning method.

Tic-Tac-Toe is a deterministic, two-player, zero-sum game on a $3\times3$ grid.

## State

A state contains:

- the complete board configuration,
- the player whose turn it is.

The board encoding is:

- $1$ = `X`,
- $-1$ = `O`,
- $0$ = empty.

## Action

An action is the index of an empty cell.

```text
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

## Transition

A legal action deterministically creates the next board state.

Therefore:

```math
P(s'|s,a)=1
```

for the state produced by the selected legal move.

## Reward

Terminal outcomes are represented as:

```math
R=
\begin{cases}
+1, & \text{win}\\
0, & \text{draw}\\
-1, & \text{loss}
\end{cases}
```

## What MCTS Estimates

MCTS does not train a neural network or learn a permanent environment model.

During search it accumulates tree statistics:

- visit count $N(s,a)$,
- accumulated return $W(s,a)$,
- mean return $\bar Q(s,a)$.

The mean return is:

```math
\bar Q(s,a)
=
\frac{W(s,a)}{N(s,a)}.
```

---

# Monte Carlo Tree Search Algorithm

One MCTS iteration contains four stages:

```text
Selection
    ↓
Expansion
    ↓
Simulation
    ↓
Backpropagation
```

## 1. Selection

The notebook uses Upper Confidence Bounds for Trees:

```math
\text{UCT}(s,a)
=
\bar Q(s,a)
+
c
\sqrt{
\frac{\ln N(s)}
{N(s,a)}
}.
```

The selected branch is:

```math
a_{\text{UCT}}
=
\arg\max_a
\text{UCT}(s,a).
```

The first term favors actions with strong observed returns. The second term encourages exploration of less-visited actions.

## 2. Expansion

If the selected node has an untried legal action, one new child is added to the tree.

## 3. Simulation

The game is rolled out from the expanded node until a terminal state is reached.

The notebook uses random legal rollout actions.

## 4. Backpropagation

The terminal result is propagated back through the selected path:

```math
N \leftarrow N+1
```

and

```math
W \leftarrow W+G.
```

The updated mean return becomes:

```math
\bar Q=\frac{W}{N}.
```

After the simulation budget is exhausted, the implementation chooses the most-visited root action:

```math
a^*
=
\arg\max_a N(s,a).
```

---

# MCTS Demonstration

The main tactical position is:

```text
X | O | X
---------
  | O |
---------
  |   |
```

It is `X`'s turn.

The legal actions are $3,5,6,7,8$.

After $2000$ search iterations, the executed notebook produces:

| Action | Visits | Mean Value |
|---:|---:|---:|
| 3 | 26 | -0.692 |
| 5 | 27 | -0.667 |
| 6 | 27 | -0.667 |
| 7 | 1896 | 0.001 |
| 8 | 24 | -0.708 |

The selected action is cell $7$.

The notebook also shows how the selected branch becomes increasingly dominant as the search budget increases from $25$ to $100$, $500$, and $2000$ simulations.

Against a random opponent using $300$ MCTS iterations per move and the notebook's fixed evaluation seeds:

- MCTS as `X`: $99$ wins, $1$ draw, $0$ losses in $100$ games,
- MCTS as `O`: $92$ wins, $8$ draws, $0$ losses in $100$ games.

These values describe this implementation and evaluation configuration rather than a universal MCTS performance guarantee.

---

# Dynamic Programming vs. Monte Carlo Tree Search

Both approaches perform planning, but they organize computation differently.

| Feature | Dynamic Programming | Monte Carlo Tree Search |
|---|---|---|
| Environment model | Known transition/reward model | Known or simulatable model |
| Main computation | Bellman backups | Tree search and rollouts |
| State coverage | Systematic sweeps | Selective search |
| Main statistics | $V(s)$ and $Q(s,a)$ | $N(s,a)$, $W(s,a)$, $\bar Q(s,a)$ |
| Exploration mechanism | Not required for exact DP backups | UCT exploration term |
| Decision process | Optimize values across the MDP | Search from the current state |
| Examples here | Gridworld, FrozenLake | Tic-Tac-Toe |

Dynamic Programming is effective when a manageable finite model can be swept repeatedly.

MCTS is useful when selectively simulating promising future trajectories is more practical than exhaustively evaluating the complete state space.

---

# Learning Progression

```text
Known MDP
   ↓
Policy Evaluation
   ↓
Policy Iteration
   ↓
Value Iteration
   ↓
Search-Based Planning
   ↓
Monte Carlo Tree Search
```

This sequence establishes how decisions can be optimized when the environment dynamics are available.

The next natural step is model-free Reinforcement Learning, where the transition probabilities and reward model are not assumed to be known in advance.

---

# Running the Notebooks

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

The notebooks are also compatible with Google Colab.

Recommended order:

1. `01. 3x4grid_value_policy_iteration_step_by_step.ipynb`
2. `02. frozenlake_dynamic_programming.ipynb`
3. `03. tic_tac_toe_monte_carlo_tree_search.ipynb`

The first notebook makes Dynamic Programming calculations transparent. The second applies the methods to a stochastic MDP. The third introduces selective search-based planning with Monte Carlo Tree Search.
