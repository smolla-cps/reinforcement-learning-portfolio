# Dynamic Programming for MDPs and Model-Based Planning

This folder demonstrates planning methods for environments whose dynamics are known or can be simulated.

It covers two main approaches:

- **Dynamic Programming:** Policy Evaluation, Policy Iteration, and Value Iteration
- **Search-Based Planning:** Monte Carlo Tree Search (MCTS)

```text
Known or simulatable environment model
                ↓
             Planning
          ┌─────┴─────┐
          ↓           ↓
Dynamic Programming   Tree Search
          ↓           ↓
Policy Evaluation     MCTS
Policy Iteration
Value Iteration
```

## Project Structure

```text
02-dynamic-programming-and-planning/
│
├── 01. 3x4grid_value_policy_iteration_step_by_step.ipynb
├── 02. frozenlake_dynamic_programming.ipynb
├── 03_deterministic_gridworld_value_vs_policy_iteration.ipynb
├── 04_tic_tac_toe_monte_carlo_tree_search.ipynb
├── README.md
└── requirements.txt
```

## 1. 3×4 GridWorld: Value Iteration and Policy Iteration

A small stochastic GridWorld is used to make Dynamic Programming calculations easy to inspect.

The notebook demonstrates:

- stochastic transitions,
- Bellman updates,
- Value Iteration,
- Policy Evaluation,
- Policy Improvement,
- Policy Iteration,
- and comparison of final values and policies.

$$
Q(s,a)
=
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V(s')
\right].
$$

## 2. Slippery FrozenLake: Dynamic Programming

A $4\times4$ slippery FrozenLake environment extends the same ideas to a standard stochastic benchmark.

```text
S F F F
F H F H
F F F H
H F F G
```

The notebook includes the transition model, Policy Evaluation, Policy Iteration, Value Iteration, convergence tracking, greedy-policy extraction, and simulation-based evaluation.

### Policy Evaluation

$$
V_{k+1}^{\pi}(s)=\sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V_k^{\pi}(s')\right]
$$

### Policy Improvement

$$
\pi_{\text{new}}(s)=\arg\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V^{\pi}(s')\right]
$$

### Value Iteration

$$
V_{k+1}(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V_k(s')\right]
$$


## 3. 7×7 Deterministic GridWorld: Value Iteration vs. Policy Iteration

This notebook applies the general Dynamic Programming algorithms to a larger deterministic GridWorld.

The algorithms are written in stochastic form, but for this environment:

$$
P(s'|s,a)=1
$$

for the single next state produced by an action.

The notebook demonstrates step-by-step Bellman calculations, Value Iteration convergence, Policy Evaluation, Policy Improvement, Policy Iteration convergence, final policies, and agent simulations for both methods.

## 4. Tic-Tac-Toe: Monte Carlo Tree Search

The final notebook introduces MCTS as a search-based planning method.

Each MCTS iteration performs:

```text
Selection
    ↓
Expansion
    ↓
Simulation
    ↓
Backpropagation
```

Selection uses:

$$
\operatorname{UCT}(s,a)
=
\bar Q(s,a)
+
c
\sqrt{
\frac{\ln N(s)}
{N(s,a)}
}.
$$

MCTS stores visit counts $N(s,a)$, accumulated returns $W(s,a)$, and mean returns $\bar Q(s,a)$.

The notebook prints each MCTS stage, shows code-generated UCT calculations, visualizes search convergence and the search tree, and simulates a complete Tic-Tac-Toe game.

## Dynamic Programming vs. MCTS

| Feature | Dynamic Programming | Monte Carlo Tree Search |
|---|---|---|
| Model | Known transition/reward model | Known or simulatable model |
| Main computation | Bellman backups | Tree search and rollouts |
| State coverage | Systematic | Selective |
| Main quantities | $V(s)$, $Q(s,a)$ | $N$, $W$, $\bar Q$ |

## Running the Notebooks

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in this order:

1. `01. 3x4grid_value_policy_iteration_step_by_step.ipynb`
2. `02. frozenlake_dynamic_programming.ipynb`
3. `03_deterministic_gridworld_value_vs_policy_iteration.ipynb`
4. `04_tic_tac_toe_monte_carlo_tree_search.ipynb`

The notebooks are also compatible with Google Colab.
