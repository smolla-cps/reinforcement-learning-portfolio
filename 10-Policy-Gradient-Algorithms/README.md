# Policy Gradient Methods: REINFORCE, Actor-Critic, and PPO

This folder develops policy-gradient reinforcement learning from the basic REINFORCE algorithm to Actor-Critic and Proximal Policy Optimization (PPO). The notebooks emphasize mathematical derivation, manual calculations, policy updates, and small reproducible Python examples.

The sequence shows how the same policy-gradient idea develops from Monte Carlo returns to critic-based learning signals and finally to PPO's clipped policy update.

## Notebook sequence

### 01 — Policy Gradients and REINFORCE

**File:** `01_Policy_Gradients_REINFORCE.ipynb`

This notebook introduces policy-gradient reinforcement learning and derives the REINFORCE algorithm step by step.

Main topics:

- policy-gradient objective and expected return;
- trajectory probability and the log-derivative trick;
- derivation of the REINFORCE gradient;
- Monte Carlo return `G_t`;
- episodic REINFORCE algorithm;
- two-action policy using sigmoid;
- four-action policy using softmax;
- derivation of the softmax log-policy gradient;
- manual policy updates for multiple episodes;
- expected-return calculations;
- stochastic REINFORCE training;
- policy-improvement plots;
- connection between REINFORCE, Vanilla Policy Gradient, Actor-Critic, and PPO.

---

### 02 — REINFORCE, Vanilla, Advantage, TD-Residual, and Q-Weighted Policy Gradient

**File:** `02_REINFORCE_Vanilla_Advantage_TD_Q_Policy_Gradient_Methods.ipynb`

This notebook compares several policy-gradient update signals using the same sampled trajectories.

Main topics:

- four-action softmax policy;
- relationship between policy parameters `θ`, logits `z`, softmax probabilities, and one-hot action vectors;
- interpretation of expectations over trajectories;
- Monte Carlo returns;
- REINFORCE updates;
- Vanilla Policy Gradient with trajectory averaging;
- Advantage Policy Gradient;
- TD-residual Policy Gradient;
- Q-weighted Policy Gradient;
- detailed calculations for trajectories of different lengths;
- actor-update direction under positive and negative learning signals;
- comparison of the different policy-gradient estimators.

The common policy-update structure is:

```math
\text{Policy Update}
=
\text{Learning Rate}
\times
\text{Learning Signal}
\times
\nabla_{\theta}\log\pi_{\theta}(A_t\mid S_t)
```

The learning signal changes by method:

```math
G_t,\qquad
Q(S_t,A_t),\qquad
Q(S_t,A_t)-V(S_t),\qquad
\delta_t
```

---

### 03 — Actor-Critic and Proximal Policy Optimization

**File:** `03_Actor_Critic_PPO_Continuous_Policies.ipynb`

This notebook extends policy gradients to one-step Actor-Critic and PPO.

Main topics:

- actor and critic roles;
- difference between REINFORCE with a baseline and Actor-Critic;
- one-step TD error;
- critic update;
- actor update;
- worked Actor-Critic trajectory;
- TD-error and action-probability plots;
- motivation for PPO;
- old-policy and new-policy probability ratio;
- PPO surrogate objective;
- clipped surrogate objective;
- effect of positive and negative advantages;
- worked PPO clipping calculations;
- PPO training loop with rollouts, epochs, and mini-batches;
- PPO as an Actor-Critic method;
- discrete-action softmax policies;
- continuous-action Gaussian policies;
- probability-density ratios for continuous actions;
- comparison of Actor-Critic and PPO.

The PPO probability ratio is:

```math
r_t(\theta)
=
\frac{
\pi_{\theta}(A_t\mid S_t)
}{
\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)
}
```

The clipped PPO objective is:

```math
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)\hat{A}_t,\;
\mathrm{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\hat{A}_t
\right)
\right]
```

## Repository structure

```text
.
├── 01_Policy_Gradients_REINFORCE.ipynb
├── 02_REINFORCE_Vanilla_Advantage_TD_Q_Policy_Gradient_Methods.ipynb
├── 03_Actor_Critic_PPO_Continuous_Policies.ipynb
├── README.md
└── requirements.txt
```

## Main concepts covered

The notebooks connect the main policy-gradient methods through their learning signals.

| Method | Main learning signal |
|---|---|
| REINFORCE | Monte Carlo return `G_t` |
| Vanilla Policy Gradient | Average of sampled trajectory gradients |
| Advantage Policy Gradient | `Q(S_t,A_t) - V(S_t)` |
| TD-Residual Policy Gradient | `R_(t+1) + γV(S_(t+1)) - V(S_t)` |
| Q-Weighted Policy Gradient | `Q(S_t,A_t)` |
| One-Step Actor-Critic | TD error `δ_t` |
| PPO | Advantage with clipped probability-ratio objective |

## Actor-Critic

Actor-Critic combines a policy learner and a value-function learner.

```math
\text{Actor-Critic}
=
\text{Actor}
+
\text{Critic}
```

The critic computes a TD error:

```math
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
```

The critic update is:

```math
w
\leftarrow
w
+
\alpha_w I_t\delta_t
\nabla_w V(S_t;w)
```

The actor update is:

```math
\theta
\leftarrow
\theta
+
\alpha_{\theta} I_t\delta_t
\nabla_{\theta}
\log\pi_{\theta}(A_t\mid S_t)
```

### Interpretation

- `θ` contains the policy or actor parameters.
- `w` contains the critic parameters.
- `δ_t` measures whether the observed transition was better or worse than the critic predicted.
- A positive `δ_t` increases the probability of the sampled action.
- A negative `δ_t` decreases the probability of the sampled action.

## Proximal Policy Optimization

PPO compares the current policy with the policy that collected the rollout data.

The probability ratio is:

```math
r_t(\theta)
=
\frac{
\pi_{\theta}(A_t\mid S_t)
}{
\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)
}
```

Interpretation:

```text
r_t = 1      → sampled-action probability did not change
r_t > 1      → sampled action became more likely
r_t < 1      → sampled action became less likely
```

With clipping parameter `ε = 0.2`:

```math
1-\epsilon=0.8,
\qquad
1+\epsilon=1.2
```

so the clipping interval is:

```math
[0.8,\;1.2]
```

The clipping function is:

```math
\mathrm{clip}(r_t,0.8,1.2)
=
\begin{cases}
0.8, & r_t < 0.8,\\
r_t, & 0.8 \le r_t \le 1.2,\\
1.2, & r_t > 1.2
\end{cases}
```

The PPO clipped objective is:

```math
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)\hat{A}_t,\;
\mathrm{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\hat{A}_t
\right)
\right]
```

The advantage estimate indicates whether the sampled action was better or worse than expected:

- `A_hat > 0` → encourage the sampled action.
- `A_hat < 0` → discourage the sampled action.

PPO clipping limits the incentive for a single optimization phase to move the policy too far from the rollout policy.

## Continuous-action policies

For discrete actions, the notebooks use softmax policies.

```math
\pi_{\theta}(a\mid s)
=
\frac{
e^{z_a(s)}
}{
\sum_{a'} e^{z_{a'}(s)}
}
```

For continuous actions, the policy can be represented by a probability density.

A common Gaussian policy is:

```math
\pi_{\theta}(a\mid s)
=
\mathcal{N}
\left(
\mu_{\theta}(s),
\sigma_{\theta}^{2}(s)
\right)
```

Its probability density is:

```math
\pi_{\theta}(a\mid s)
=
\frac{
1
}{
\sigma_{\theta}(s)\sqrt{2\pi}
}
\exp
\left[
-\frac{
\left(a-\mu_{\theta}(s)\right)^2
}{
2\sigma_{\theta}^{2}(s)
}
\right]
```

Instead of assigning a separate probability to every possible continuous action, the network learns distribution parameters such as:

```math
\mu_{\theta}(s)
\qquad\text{and}\qquad
\sigma_{\theta}(s)
```

PPO can then compute the same old-to-new policy ratio using probability densities.

## Conceptual progression

```text
REINFORCE
    ↓
Advantage / TD / Q-based policy gradients
    ↓
Actor-Critic
    ↓
PPO
```

Mathematically:

```math
\text{REINFORCE}
\rightarrow
\text{Advantage / TD / Q-based updates}
\rightarrow
\text{Actor-Critic}
\rightarrow
\text{PPO}
```

## Python libraries

The notebooks use:

- NumPy
- Pandas
- Matplotlib
- IPython/Jupyter

No deep-learning framework is required for the manual examples in this folder.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The notebooks can also be opened directly in Google Colab.

## Recommended order

```text
01 → Policy Gradients and REINFORCE
02 → Policy-Gradient Update Methods
03 → Actor-Critic and PPO
```

## Purpose

These notebooks are designed as a reinforcement-learning portfolio and study reference. The focus is on understanding the equations, following the numerical updates step by step, and connecting the mathematics to executable Python code.
