# Dynamic Programming

This folder contains four notebooks that develop Dynamic Programming for finite Markov Decision Processes (MDPs), from the mathematical foundations to stochastic and deterministic GridWorld applications.

The notebooks focus on Policy Evaluation, Policy Iteration, Value Iteration, and Modified Policy Iteration. Each example connects the Bellman equations to step-by-step calculations and Python implementation.

## Notebooks

### 1. Dynamic Programming Robot MDP

**Recommended filename:** `01_dynamic_programming_robot_mdp.ipynb`

This notebook introduces the main Dynamic Programming concepts for a finite MDP.

Topics include:

- state-value and action-value functions
- Bellman expectation equation
- Bellman optimality equation
- iterative policy evaluation
- Policy Iteration
- Value Iteration
- Modified Policy Iteration
- stochastic state transitions
- policy improvement
- convergence
- comparison of the three Dynamic Programming methods

The example uses an eight-state robot-navigation MDP with two actions, `Left` and `Right`, stochastic transition probabilities, terminal states, state rewards, and a discount factor.

This notebook is the main theoretical introduction for the folder.

---

### 2. 3×4 Stochastic GridWorld

**Notebook:** `02. 3x4grid_value_policy_iteration_step_by_step.ipynb`

This notebook applies Value Iteration and Policy Iteration to a small stochastic GridWorld.

Topics include:

- stochastic movement
- Bellman backups
- synchronous value updates
- manual state-value calculations
- Value Iteration sweeps
- Policy Evaluation
- Policy Improvement
- complete Policy Iteration
- comparison of final values and policies

The notebook is designed to show how values propagate through a stochastic spatial environment step by step.

---

### 3. Slippery FrozenLake

**Notebook:** `03. frozenlake_dynamic_programming.ipynb`

This notebook applies Dynamic Programming to the standard 4×4 slippery FrozenLake problem.

The environment contains:

- 16 states
- 4 actions: Left, Down, Right, and Up
- stochastic transitions
- holes as terminal failure states
- a goal as the terminal success state
- reward received when the transition enters the goal

Topics include:

- Policy Evaluation for a stochastic policy
- Policy Iteration
- Value Iteration
- Bellman expectation and optimality updates
- convergence
- greedy policy extraction
- comparison of Policy Iteration and Value Iteration
- simulation-based evaluation of the final policies

The FrozenLake transition model is implemented directly in the notebook, so Gymnasium is not required.

---

### 4. 7×7 Deterministic GridWorld

**Notebook:** `04_deterministic_gridworld_value_vs_policy_iteration.ipynb`

This notebook applies Value Iteration and Policy Iteration to a larger deterministic 7×7 GridWorld.

Topics include:

- state, action, transition, and reward definitions
- expected action values
- Value Iteration
- Bellman sweeps
- value propagation
- greedy policy extraction
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- convergence plots
- policy simulation
- comparison of Value Iteration and Policy Iteration

This example shows how the same Dynamic Programming ideas extend from small examples to a larger state space.

---

## Learning Progression

The notebooks are arranged in the following order:

```text
1. Robot MDP
   Learn the equations and algorithms
            ↓
2. 3×4 Stochastic GridWorld
   Follow Bellman updates step by step
            ↓
3. Slippery FrozenLake
   Apply Dynamic Programming to a stochastic benchmark
            ↓
4. 7×7 Deterministic GridWorld
   Apply the methods to a larger custom environment
```

## Core Dynamic Programming Equations

### Policy Evaluation

For a stochastic policy,

$$
V_{k+1}^{\pi}(s)
=
\sum_a \pi(a|s)
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V_k^{\pi}(s')
\right].
$$

For a deterministic policy,

$$
V_{k+1}^{\pi}(s)
=
\sum_{s',r}
p(s',r|s,\pi(s))
\left[
r+\gamma V_k^{\pi}(s')
\right].
$$

### Policy Improvement

$$
\pi_{\text{new}}(s)
=
\arg\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V^{\pi}(s')
\right].
$$

### Value Iteration

$$
V_{k+1}(s)
=
\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V_k(s')
\right].
$$

After convergence, the greedy policy is

$$
\pi^*(s)
=
\arg\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V^*(s')
\right].
$$

## Policy Iteration vs. Value Iteration

**Policy Iteration** alternates between evaluating the current policy and improving it.

```text
Policy Evaluation → Policy Improvement → Policy Evaluation → ... → Optimal Policy
```

**Value Iteration** combines evaluation and improvement through the Bellman optimality update.

```text
Bellman Optimality Updates → Converged Values → Greedy Optimal Policy
```

**Modified Policy Iteration** lies between the two approaches by performing only a limited number of Policy Evaluation sweeps before each Policy Improvement step.

## Requirements

The notebooks use Python and common scientific-computing libraries.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:

- NumPy
- Matplotlib
- Pandas
- NetworkX
- IPython
- Jupyter

Python 3.10 or later is recommended.

## Running the Notebooks

Clone the repository and move into this folder:

```bash
git clone <repository-url>
cd <repository-name>/Dynamic_Programming
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

The notebooks can also be opened directly in Google Colab.

## Repository Structure

```text
Dynamic_Programming/
│
├── README.md
├── requirements.txt
├── 01_dynamic_programming_robot_mdp.ipynb
├── 02. 3x4grid_value_policy_iteration_step_by_step.ipynb
├── 03. frozenlake_dynamic_programming.ipynb
└── 04_deterministic_gridworld_value_vs_policy_iteration.ipynb
```

## Main Concepts Covered

- finite Markov Decision Processes
- Dynamic Programming
- Bellman expectation equation
- Bellman optimality equation
- state-value functions
- action-value functions
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
- Modified Policy Iteration
- stochastic transitions
- deterministic transitions
- convergence
- greedy policy extraction
