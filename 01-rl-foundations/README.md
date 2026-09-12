# Reinforcement Learning Foundations

This section develops the core ideas needed to understand reinforcement learning before moving to planning, tabular learning, deep reinforcement learning, and policy optimization.

A common 7×7 GridWorld is used across the notebooks so that the mathematical ideas can be compared without changing the underlying problem. The material moves from the general Markov Decision Process framework to deterministic and stochastic environments, then examines how different reward criteria change the objective.

## Learning Progression

```text
Markov Decision Process
        ↓
State, Action, Transition, Reward, Policy
        ↓
Deterministic GridWorld
        ↓
Vπ(s), Qπ(s,a), Bellman Equations
        ↓
Stochastic GridWorld
        ↓
Probability-Weighted Transitions and Expectations
        ↓
Reward Criteria
        ↓
Total Reward, Average Reward, Discounted Reward
```

## GridWorld

The notebooks use a 7×7 GridWorld with the following structure:

- **Start state:** `(0, 0)`
- **Goal state:** `(6, 6)`
- **Obstacles:** `(2, 2)`, `(3, 3)`, `(4, 4)`, `(5, 2)`, `(1, 5)`
- **Actions:** `Up`, `Down`, `Left`, `Right`
- **Regular movement reward:** `-1`
- **Obstacle attempt:** remain in the current state and receive `-5`
- **Goal reward:** `+10`
- **Boundary attempt:** remain in the current state and receive `-1`

Obstacle cells are excluded from the valid state space.

The environment is represented as a Markov Decision Process:

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)
$$

where:

* $\mathcal{S}$ is the state space,
* $\mathcal{A}$ is the action space,
* $P(s'|s,a)$ is the transition model,
* $R(s,a,s')$ is the reward model,
* $\gamma$ is the discount factor.


---

## 01. Markov Decision Processes

**Notebook:** `01_markov_decision_processes.ipynb`

This notebook develops the MDP framework in detail and provides the mathematical basis for the remaining notebooks.

Topics include:

- agent–environment interaction,
- the Markov property,
- the MDP tuple,
- state and action spaces,
- deterministic and stochastic transition models,
- reward models,
- initial-state distributions,
- terminal states,
- episodic and continuing tasks,
- deterministic and stochastic policies,
- policy randomness and transition randomness,
- trajectories,
- return and discount factor,
- state-value function \(V^\pi(s)\),
- action-value function \(Q^\pi(s,a)\),
- Bellman expectation equations,
- optimal value functions,
- optimal policies,
- transition matrices,
- expected reward vectors,
- policy-induced Markov Reward Processes,
- deterministic vs. stochastic MDPs,
- model-based vs. model-free methods,
- MDPs vs. POMDPs,
- practical MDP design.

A finite MDP is represented by

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma).
\]

The notebook also demonstrates how a fixed policy induces a Markov Reward Process.

---

## 02. Deterministic Reinforcement Learning Foundations

**Notebook:** `02_rl_foundations_deterministic_gridworld.ipynb`

This notebook studies the GridWorld as a deterministic MDP.

For a given state-action pair, one next state occurs with probability 1:

\[
P(s'|s,a)=1
\]

for the realized next state.

The notebook demonstrates:

- deterministic transitions,
- boundary and obstacle behavior,
- the Markov property,
- agent–environment interaction,
- policies,
- returns and discounting,
- \(V^\pi(s)\),
- \(Q^\pi(s,a)\),
- Bellman expectation equation,
- Bellman optimality equation,
- exploration vs. exploitation,
- \(\epsilon\)-greedy action selection.

The numerical examples are obtained directly from the GridWorld model.

---

## 03. Stochastic Reinforcement Learning Foundations

**Notebook:** `03_rl_foundations_stochastic_gridworld.ipynb`

This notebook extends the same GridWorld to a stochastic transition model.

For each intended action:

- intended movement occurs with probability `0.80`,
- one perpendicular slip occurs with probability `0.10`,
- the other perpendicular slip occurs with probability `0.10`.

Therefore, a state-action pair may lead to several next states:

\[
0 \leq P(s'|s,a) \leq 1,
\qquad
\sum_{s'}P(s'|s,a)=1.
\]

The notebook demonstrates:

- transition probability distributions,
- expected immediate reward,
- empirical transition frequencies,
- the Markov property under uncertainty,
- policy stochasticity vs. environment stochasticity,
- stochastic trajectories,
- expected returns,
- \(V^\pi(s)\) under stochastic transitions,
- probability-weighted \(Q^\pi(s,a)\),
- Bellman expectation equation in its full stochastic form,
- Bellman optimality using expected action values,
- exploration vs. environmental randomness.

For a stochastic MDP,

\[
Q^\pi(s,a)
=
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^\pi(s')
\right].
\]

---

## 04. Reward Types and Criteria

**Notebook:** `04_reward_type_and_criteria.ipynb`

This notebook separates the **immediate reward function** from the rule used to evaluate a sequence of rewards.

It compares three common criteria using trajectories generated by the same GridWorld.

### Total Reward

\[
G_{\text{total}}
=
\sum_{t=1}^{T}R_t
\]

Useful for finite episodic tasks when the total reward remains finite.

### Average Reward

\[
\rho
=
\lim_{n\rightarrow\infty}
\frac{1}{n}
\sum_{t=1}^{n}R_t
\]

Useful for continuing systems where long-run reward per time step is the main objective.

### Discounted Reward

\[
G_t
=
\sum_{k=0}^{\infty}
\gamma^kR_{t+k+1},
\qquad
0\leq\gamma<1
\]

Useful when future rewards should receive less weight or when a finite return is needed for an infinite-horizon task.

The notebook also compares:

- short and long paths,
- finite episode-average reward vs. long-run average reward,
- episodic and continuing versions of the GridWorld,
- different discount factors,
- earlier vs. later goal rewards,
- convergence of running average reward,
- selection of a reward criterion based on the task.

For the original episodic GridWorld, total reward or discounted reward is natural. For a continuing version of the same system, average reward becomes a meaningful long-run objective.

---

## Project Structure

```text
01-rl-foundations/
│
├── README.md
├── requirements.txt
├── 01_markov_decision_processes.ipynb
├── 02_rl_foundations_deterministic_gridworld.ipynb
├── 03_rl_foundations_stochastic_gridworld.ipynb
└── 04_reward_type_and_criteria.ipynb
```

## Recommended Order

Run the notebooks in numerical order:

1. `01_markov_decision_processes.ipynb`
2. `02_rl_foundations_deterministic_gridworld.ipynb`
3. `03_rl_foundations_stochastic_gridworld.ipynb`
4. `04_reward_type_and_criteria.ipynb`

This order moves from the general mathematical framework to concrete deterministic and stochastic examples, then to the choice of reward criterion.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in numerical order.

## Dependencies

The notebooks use Python standard-library modules together with:

- NumPy
- Matplotlib
- Jupyter
