# Planning and Dynamic Programming

This section demonstrates how a finite Markov Decision Process can be solved when the environment model is known.

The complete transition model $P(s'|s,a)$ and reward model $R(s,a,s')$ are available, so the solution can be computed with Dynamic Programming rather than learned from sampled experience.

The progression is:

**Policy Evaluation → Policy Iteration → Value Iteration → Algorithm Comparison**

Two environments are used. A small 3×4 Gridworld makes the Bellman updates and numerical calculations easy to follow, while slippery FrozenLake provides a standard stochastic planning benchmark.

---

## Project Structure

```text
02-planning-and-dynamic-programming/
│
├── README.md
├── requirements.txt
├── 01. 3x4grid_value_policy_iteration_step_by_step.ipynb
└── 02. frozenlake_dynamic_programming.ipynb
```

### 1. 3×4 Gridworld: Value Iteration and Policy Iteration Step by Step

This notebook focuses on the numerical mechanics of Dynamic Programming.

It shows:

- the finite MDP and stochastic transition model,
- Bellman optimality updates,
- synchronous Value Iteration,
- state-value propagation across iterations,
- manual calculations for individual states,
- Policy Evaluation,
- Policy Improvement,
- complete Policy Iteration,
- final optimal values and policies,
- and a direct comparison between Value Iteration and Policy Iteration.

The Gridworld uses stochastic motion:

- intended direction: $0.8$,
- perpendicular slip: $0.1$,
- opposite perpendicular slip: $0.1$.

The terminal state values are $+1$ and $-1$, and the discount factor is $\gamma=0.9$.

The converged Value Iteration solution is:

```text
 0.64   0.74   0.85   1.00
 0.57   WALL   0.57  -1.00
 0.49   0.43   0.48   0.28
```

The final greedy policy is:

```text
→   →   →   +1
↑  WALL ↑   -1
↑   ←   ↑    ←
```

Value Iteration and Policy Iteration select the same final action in all nine non-terminal states.

---

### 2. Slippery FrozenLake: Dynamic Programming

This notebook applies the same planning ideas to the standard 4×4 slippery FrozenLake benchmark.

```text
S F F F
F H F H
F F F H
H F F G
```

The notebook includes:

- the exact FrozenLake transition model,
- state and action definitions,
- stochastic transition probabilities,
- transition rewards,
- terminal-state handling,
- Policy Evaluation of a random policy,
- Bellman expectation verification,
- Policy Iteration from an initial deterministic policy,
- Value Iteration from $V_0=0$,
- numerical action-value calculations from the model,
- convergence tracking,
- greedy-policy extraction,
- comparison of final value functions and policies,
- model-backup counts,
- and simulation-based policy evaluation.

With $\gamma=0.99$ and convergence tolerance $\theta=10^{-10}$, both planning methods converge to essentially the same optimal value function.

For the executed notebook:

- Policy Iteration converges in 7 outer policy-improvement iterations.
- Value Iteration converges in 571 Bellman optimality sweeps.
- The maximum difference between the final value functions is approximately $1.3\times10^{-11}$.
- The two final policies have complete action agreement.
- In 3,000 stochastic simulation episodes, both final policies achieve a success rate of approximately $0.749$ under the notebook's evaluation settings.

---

## Dynamic Programming Equations

### Policy Evaluation

Policy Evaluation answers:

> What is the expected long-term value of each state if the current policy continues to be followed?

For a stochastic policy $\pi(a|s)$:

```math
V_{k+1}^{\pi}(s)=\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s')+\gamma V_k^{\pi}(s')\right]
```

There is no $\max$ over actions during Policy Evaluation. The current policy determines which actions are evaluated.

For a deterministic policy:

```math
V_{k+1}^{\pi}(s)=\sum_{s'} P(s'|s,\pi(s)) \left[R(s,\pi(s),s')+\gamma V_k^{\pi}(s')\right]
```

---

### Policy Improvement

After $V^{\pi}$ has converged, all actions are compared:

```math
Q^{\pi}(s,a)=\sum_{s'} P(s'|s,a) \left[R(s,a,s')+\gamma V^{\pi}(s')\right]
```

The improved action is:

```math
\pi_{\text{new}}(s)=\arg\max_a Q^{\pi}(s,a)
```

Policy Iteration repeatedly performs:

```text
Current Policy
      ↓
Policy Evaluation
      ↓
V^π converges
      ↓
Policy Improvement
      ↓
argmax over actions
      ↓
New Policy
      ↓
Repeat until the policy is unchanged
```

---

### Value Iteration

Value Iteration applies the Bellman optimality update directly:

```math
V_{k+1}(s)=\max_a \sum_{s'} P(s'|s,a) \left[R(s,a,s')+\gamma V_k(s')\right]
```

At each state:

```text
Calculate Q(s, Left)
Calculate Q(s, Down)
Calculate Q(s, Right)
Calculate Q(s, Up)
        ↓
Take max
        ↓
Store the new state value
```

The $\max$ returns the best numerical value.

After convergence, the final greedy policy is extracted using:

```math
\pi^*(s)=\arg\max_a \sum_{s'} P(s'|s,a) \left[R(s,a,s')+\gamma V^*(s')\right]
```

The $\arg\max$ returns the action that produces the largest expected value.

---

## Policy Iteration vs. Value Iteration

| Feature | Policy Iteration | Value Iteration |
|---|---|---|
| Maintains an explicit policy during planning | Yes | Not required during value updates |
| Evaluates a fixed policy | Yes | No |
| Uses $\max_a$ during every value update | No | Yes |
| Uses $\arg\max_a$ | During policy improvement | After convergence to extract the greedy policy |
| Inner convergence | $V^{\pi}$ for the current policy | $V_k$ toward $V^*$ |
| Final objective | Optimal policy and value function | Optimal value function and greedy optimal policy |

Both methods require the environment model to be known.

---

## Reward Conventions

The two notebooks intentionally illustrate two common reward conventions.

### 3×4 Gridworld

The terminal cells are assigned fixed values $+1$ and $-1$.

### FrozenLake

Reward is attached to the transition:

```math
R(s,a,s')=1
```

when the transition enters the goal, and $0$ otherwise.

Because the reward definitions differ, equations should always be interpreted together with the environment's reward convention.

---

## Planning with a Known Model

These notebooks demonstrate **planning**, not model-free trial-and-error learning.

The algorithms already know:

- the state space $\mathcal{S}$,
- the action space $\mathcal{A}$,
- the transition probabilities $P(s'|s,a)$,
- the reward model $R(s,a,s')$.

They use this model repeatedly to calculate expected values and improve decisions.

This provides the foundation for the later question:

> What happens when $P(s'|s,a)$ and $R(s,a,s')$ are not known?

That leads naturally to model-free Reinforcement Learning methods.

---

## Running the Notebooks

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

The notebooks are also compatible with Google Colab.

Run the notebooks in this order:

1. `01. 3x4grid_value_policy_iteration_step_by_step.ipynb`
2. `02. frozenlake_dynamic_programming.ipynb`

The first notebook emphasizes transparent mathematical calculation. The second applies the same ideas to a stochastic benchmark and performs a complete algorithm comparison.
