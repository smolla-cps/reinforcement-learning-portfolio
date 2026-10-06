# Policy Gradient Methods: REINFORCE, Actor-Critic, and PPO

This folder develops policy-gradient reinforcement learning from the basic REINFORCE algorithm to Actor-Critic and Proximal Policy Optimization (PPO). The notebooks emphasize mathematical derivation, manual calculations, policy updates, and small reproducible Python examples.

The sequence is designed to show how the same policy-gradient idea develops from Monte Carlo returns to critic-based learning signals and finally to PPO's clipped policy update.

## Notebook sequence

### 01 — Policy Gradients and REINFORCE

**File:** `01_Policy_Gradients_REINFORCE.ipynb`

This notebook introduces policy-gradient reinforcement learning and derives the REINFORCE algorithm step by step.

Main topics:

- policy-gradient objective and expected return;
- trajectory probability and the log-derivative trick;
- derivation of the REINFORCE gradient;
- Monte Carlo return \(G_t\);
- episodic REINFORCE algorithm;
- two-action policy using sigmoid;
- four-action policy using softmax;
- derivation of the softmax log-policy gradient;
- manual policy updates for multiple episodes;
- expected-return calculations;
- stochastic REINFORCE training;
- policy-improvement plots;
- connection between REINFORCE, Vanilla Policy Gradient, Actor-Critic, and PPO.

The notebook uses both two-action and four-action examples so that the scalar and vector forms of the policy gradient can be compared directly.

---

### 02 — REINFORCE, Vanilla, Advantage, TD-Residual, and Q-Weighted Policy Gradient

**File:** `02_REINFORCE_Vanilla_Advantage_TD_Q_Policy_Gradient_Methods.ipynb`

This notebook compares several policy-gradient update signals using the same sampled trajectories.

Main topics:

- four-action softmax policy;
- relationship between policy parameters \(\theta\), logits \(z\), softmax probabilities, and one-hot action vectors;
- interpretation of expectations over trajectories;
- Monte Carlo returns;
- REINFORCE updates;
- Vanilla Policy Gradient with trajectory averaging;
- Advantage Policy Gradient;
- TD-residual Policy Gradient;
- Q-weighted Policy Gradient;
- detailed calculations for trajectories of different lengths;
- actor update direction under positive and negative learning signals;
- comparison of the different policy-gradient estimators.

The central structure is

\[
\text{policy update}
=
\text{learning rate}
\times
\text{learning signal}
\times
\nabla_\theta \log \pi_\theta(A_t\mid S_t).
\]

The learning signal changes by method:

\[
G_t,\qquad
Q(S_t,A_t),\qquad
Q(S_t,A_t)-V(S_t),\qquad
\delta_t.
\]

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

The PPO probability ratio is

\[
r_t(\theta)
=
\frac{\pi_\theta(A_t\mid S_t)}
{\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)}.
\]

The clipped objective is

\[
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E
\left[
\min
\left(
r_t(\theta)\hat A_t,\;
\operatorname{clip}
\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right].
\]

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
| REINFORCE | Monte Carlo return \(G_t\) |
| Vanilla Policy Gradient | Average of sampled trajectory gradients |
| Advantage Policy Gradient | \(Q(S_t,A_t)-V(S_t)\) |
| TD-Residual Policy Gradient | \(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\) |
| Q-Weighted Policy Gradient | \(Q(S_t,A_t)\) |
| One-Step Actor-Critic | TD error \(\delta_t\) |
| PPO | Advantage with clipped probability-ratio objective |

## Actor-Critic idea

Actor-Critic combines a policy learner and a value-function learner.

The critic computes a TD error:

\[
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t).
\]

The critic update is

\[
w
\leftarrow
w
+
\alpha_w I_t\delta_t
\nabla_w V(S_t;w),
\]

and the actor update is

\[
\theta
\leftarrow
\theta
+
\alpha_\theta I_t\delta_t
\nabla_\theta
\log \pi_\theta(A_t\mid S_t).
\]

## PPO idea

PPO compares the new policy with the policy that collected the rollout data.

If

\[
r_t(\theta)=1,
\]

the sampled action probability has not changed.

If

\[
r_t(\theta)>1,
\]

the action became more likely.

If

\[
r_t(\theta)<1,
\]

the action became less likely.

With a clipping parameter such as \(\epsilon=0.2\), PPO limits the useful ratio range to approximately

\[
[0.8,1.2].
\]

This reduces the incentive for a single optimization step to move the policy too far from the rollout policy.

## Continuous actions

For discrete actions, the notebooks use softmax policies.

For continuous actions, the policy can be represented by a probability density such as

\[
\pi_\theta(a\mid s)
=
\mathcal N
\left(
\mu_\theta(s),
\sigma_\theta^2(s)
\right).
\]

Instead of assigning one probability to every possible action, the policy network learns the parameters of the action distribution.

## Python libraries

The notebooks use:

- NumPy
- Pandas
- Matplotlib
- IPython/Jupyter display utilities

No deep-learning framework is required for the manual examples in this folder.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

The notebooks can also be opened directly in Google Colab.

## Recommended order

Run the notebooks in this order:

```text
01 → Policy Gradient and REINFORCE
02 → Policy-gradient update variants
03 → Actor-Critic and PPO
```

This order follows the conceptual progression:

\[
\boxed{
\text{REINFORCE}
\rightarrow
\text{Advantage / TD / Q-based updates}
\rightarrow
\text{Actor-Critic}
\rightarrow
\text{PPO}
}
\]

## Purpose

These notebooks are written as a reinforcement-learning portfolio and study reference. The focus is on understanding the equations, seeing the numerical updates explicitly, and connecting the mathematics to executable Python code.
