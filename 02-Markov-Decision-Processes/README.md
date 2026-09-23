<!-- Math note: equations are intentionally written with single-dollar inline LaTeX delimiters for reliable GitHub README rendering. -->

# Markov Decision Processes

This folder develops the mathematical and computational foundations of **Markov Decision Processes (MDPs)** and shows how the same framework behaves in deterministic and stochastic environments.

The notebooks progress from the formal definition of an MDP to value functions, Bellman equations, optimal policies, and GridWorld implementations.

## Contents

### 01. Finite Markov Decision Processes

**Notebook:** `01_markov_decision_processes.ipynb`

Introduces the main theoretical foundations of finite MDPs, including:

- stochastic processes and the Markov property
- stationary transition dynamics
- transition probabilities and transition matrices
- the agent–environment interaction
- the MDP tuple
- states, actions, rewards, transitions, and terminal conditions
- episodic and continuing tasks
- return and discounting
- policies and trajectories
- state-value and action-value functions
- the relationship between $V^\pi(s)$ and $Q^\pi(s,a)$
- Bellman expectation equations
- Bellman optimality equations
- optimal value functions and optimal policies
- model-based vs. model-free methods
- fully observable MDPs vs. POMDPs

Worked examples include the **Recycling Robot**, **Pole Balancing**, and **GridWorld**.

### 02. Deterministic vs. Stochastic GridWorld

**Notebook:** `02_markov_deterministic_stochastic_gridworld.ipynb`

Applies the MDP framework to a shared **7×7 GridWorld** and compares deterministic and stochastic transitions.

Topics include:

- deterministic transition dynamics
- stochastic transition dynamics
- Markov property
- policy randomness vs. environment randomness
- agent trajectories
- discounted return
- $V^\pi(s)$ and $Q^\pi(s,a)$
- Bellman expectation equations
- optimal value functions
- Bellman optimality equations
- value iteration
- optimal-policy visualization
- exploration vs. exploitation vs. environmental randomness

The same environment structure is used in both cases so that the effect of transition uncertainty can be compared directly.

## Core Mathematical Relationships

### MDP dynamics

$p(s',r\mid s,a) = P(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a)$

### Return

$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

### State-value function

$V^\pi(s) = \mathbb{E}_\pi[G_t\mid S_t=s]$

### Action-value function

$Q^\pi(s,a) = \mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$

### Relationship between state and action values

$V^\pi(s) = \sum_a \pi(a\mid s)Q^\pi(s,a)$

### Bellman expectation equation

$V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a) \left[ r+\gamma V^\pi(s') \right]$

### Bellman optimality equation

```math
V^*(s) =
\max_a \sum_{s',r} p(s',r \mid s,a)
\left[ r + \gamma V^*(s') \right]
```

## Repository Structure

```text
02-Markov-Decision-Processes/
│
├── README.md
├── requirements.txt
├── 01_markov_decision_processes.ipynb
└── 02_markov_deterministic_stochastic_gridworld.ipynb
```

## Requirements

The notebooks use:

- Python 3
- NumPy
- pandas
- Matplotlib

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

Clone the repository, open this folder, and launch the notebooks in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

Run the notebooks in numerical order:

1. `01_markov_decision_processes.ipynb`
2. `02_markov_deterministic_stochastic_gridworld.ipynb`

The first notebook develops the MDP theory, while the second applies those ideas to deterministic and stochastic GridWorld environments.

## Learning Progression

The notebooks follow this progression:

```math
\boxed{
\text{Markov Property}
\rightarrow
\text{MDP}
\rightarrow
\text{Policy}
\rightarrow
V^\pi
\rightarrow
Q^\pi
\rightarrow
\text{Bellman Equations}
\rightarrow
V^*, Q^*
\rightarrow
\text{Optimal Policy}
}
```

This provides the foundation for later reinforcement-learning methods such as Dynamic Programming, Monte Carlo methods, Temporal-Difference learning, SARSA, and Q-learning.
