# Temporal-Difference Learning

This repository presents a detailed implementation of **Temporal-Difference (TD) learning** for reinforcement-learning prediction problems. The notebook develops the ideas step by step, starting from the relationship between Monte Carlo, Dynamic Programming, and TD learning, and then moving to TD(0), TD(1), TD($\lambda$), eligibility traces, and empirical evaluation on the classic five-state Random Walk problem.

The notebook combines mathematical derivations, hand-worked episode calculations, algorithms, Python implementations, and learning-curve visualizations.

## Notebook

`Temporal_Difference_Learning.ipynb`

## Topics Covered

The notebook includes:

- Introduction to Temporal-Difference learning
- Relationship among Dynamic Programming, Monte Carlo, and TD
- Monte Carlo versus TD learning
- Driving Home example showing why TD can update before the final outcome
- Complete episode comparison of Monte Carlo and TD(0)
- TD(1) prediction
- TD(0) one-step prediction
- Five-state Random Walk prediction problem
- True state values for the Random Walk
- TD(0) value estimates after different numbers of episodes
- Root-Mean-Square Error (RMSE)
- Empirical RMSE comparison of TD(0) and Monte Carlo for different learning rates
- TD($\lambda$)
- Eligibility traces
- Backward-view TD($\lambda$)
- Step-by-step TD($\lambda$) calculation over a longer episode
- Bootstrapping and sampling

## Core Equations

### Monte Carlo update

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
$$

where the return is

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

Monte Carlo waits until the return is available before updating the value estimate.

### TD(0) update

$$
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right]
$$

The TD error is

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

so the update can also be written as

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t
$$

TD(0) can update after every transition because it bootstraps from the current estimate of the next state.

### Backward-view TD($\lambda$)

The TD error is

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

For accumulating eligibility traces,

$$
e_t(s)=\gamma\lambda e_{t-1}(s)+\mathbf{1}(s=S_t)
$$

and every state is updated using

$$
V(s)\leftarrow V(s)+\alpha\delta_t e_t(s)
$$

The trace-decay parameter $\lambda$ controls how strongly the current TD error is assigned to previously visited states.

## Random Walk Problem

The notebook uses the five-state Random Walk

```text
Left Terminal - A - B - C - D - E - Right Terminal
```

Each episode starts from state `C`.

At every nonterminal state,

```text
P(left)  = 0.5
P(right) = 0.5
```

All rewards are zero except the transition from `E` to the right terminal, which gives a reward of `1`.

For the undiscounted task, the true state values are

$$
v_\pi(A)=\frac{1}{6},\quad
v_\pi(B)=\frac{2}{6},\quad
v_\pi(C)=\frac{3}{6},\quad
v_\pi(D)=\frac{4}{6},\quad
v_\pi(E)=\frac{5}{6}
$$

These values represent the probability of reaching the right terminal before the left terminal from each state.

## Empirical TD(0) and Monte Carlo Comparison

The notebook compares prediction performance using RMSE:

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(\hat v_i-v_i)^2
}
$$

The experiments evaluate several constant learning rates.

TD(0):

```text
α = 0.05, 0.10, 0.15
```

Monte Carlo:

```text
α = 0.01, 0.02, 0.03, 0.04
```

For each setting, the notebook:

1. Initializes the five nonterminal state values to `0.5`.
2. Runs 100 episodes.
3. Calculates RMSE after every episode.
4. Repeats the experiment over 100 independent runs.
5. Averages the RMSE across runs.

The resulting learning curves show how the prediction error changes across episodes for different learning rates.

## TD($\lambda$) Example

The backward-view TD($\lambda$) example uses the longer episode

$$
A\xrightarrow{0}B\xrightarrow{0}C\xrightarrow{0}D\xrightarrow{0}E\xrightarrow{1}\text{Terminal}
$$

with

$$
\alpha=0.1,\qquad\gamma=1,\qquad\lambda=0.8
$$

The example shows how eligibility traces retain information about previously visited states and how a later TD error is propagated backward with decreasing strength.

## Repository Structure

```text
Temporal-Difference-Learning/
│
├── Temporal_Difference_Learning.ipynb
├── README.md
└── requirements.txt
```

## Requirements

The notebook uses:

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

### Google Colab

Upload `Temporal_Difference_Learning.ipynb` to Google Colab and run the cells from top to bottom.

### Local Jupyter

Clone or download the repository, install the requirements, and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open `Temporal_Difference_Learning.ipynb`.

## Main Python Tools

The implementation uses:

```python
import numpy as np
import matplotlib.pyplot as plt
```

NumPy is used for numerical calculations, random sampling, state-value arrays, and RMSE calculations. Matplotlib is used for the Driving Home example, Random Walk value estimates, empirical RMSE learning curves, and eligibility-trace visualizations.

## Scope

This notebook focuses on **TD prediction**, where the objective is to estimate the value function for a fixed policy.

TD control methods such as **SARSA, Q-learning, and Expected SARSA** are intentionally left for a separate notebook.
