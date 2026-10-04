# 09 — Deep Reinforcement Learning: DQN and Double DQN

A practical study of **Deep Q-Networks (DQN)** and **Double DQN**, from Bellman equations and fully worked weight updates to a trainable agent in the Gymnasium `CartPole-v1` environment. The notebook combines mathematical explanations, numerical checks, algorithm pseudocode, diagrams, PyTorch implementation, training diagnostics, and agent visualization.

## Objectives

- Explain how reinforcement learning and deep learning work together to estimate action values.
- Identify when a neural-network Q-function is preferable to a tabular Q-table.
- Understand **experience replay**: stored transitions, random sampling, and reuse of past data.
- Understand **mini-batch learning**: average loss/gradient across sampled transitions, followed by one online-network update.
- Explain why DQN uses an **online network** and a periodically synchronized **target network**.
- Derive DQN targets, TD errors, squared-error loss, and the semi-gradient update.
- Show why Double DQN separates next-action **selection** from **evaluation** to reduce maximization-related overestimation bias.
- Train, evaluate, and visualize a DQN or Double DQN agent on CartPole.

## Notebook contents

| Sections | Topics |
|---|---|
| 1–4 | Deep RL foundations, tabular Q-learning versus DQN, algorithm families, Q-network architecture |
| 5–7 | Full learning cycle, replay memory and mini-batches, online versus target networks |
| 8–10 | Bellman targets, loss and gradients, complete worked single-transition and mini-batch calculations |
| 11–13 | Epsilon-greedy exploration, DQN pseudocode, Double DQN explanation and numerical comparison |
| 14–15 | Original Atari DQN background and CartPole-v1 environment |
| 16–20 | PyTorch Q-network, replay buffer, hyperparameters, optimizer, training loop |
| 21–24 | Reward/loss/epsilon plots, random baseline, Q-value inspection, trained-agent GIF |
| 25–26 | Interpretation, limitations, key equations, and references |

## How DQN works

A transition stores the observed state, selected action, reward, next state, and **true termination** flag:

$$
(s_t, a_t, r_{t+1}, s_{t+1}, d_t).
$$

Transitions remain in replay memory when sampled; when the bounded buffer fills, new transitions replace the oldest. DQN samples random *transitions* (not necessarily full episodes) into mini-batches.

The **online network** predicts the action value for each sampled state/action pair. A separate **target network** supplies a temporarily fixed bootstrap value. For a mini-batch of size $B$:

$$
y_j^{\mathrm{DQN}} = r_j + \gamma(1-d_j)\max_{a'}Q(s'_j,a';\theta^-),
$$

$$
\hat y_j = Q(s_j,a_j;\theta),
\qquad
L(\theta) = \frac{1}{2B}\sum_{j=1}^{B}(y_j^{\mathrm{DQN}}-\hat y_j)^2.
$$

Backpropagation updates **only** the online-network parameters $\theta$. The target-network parameters $\theta^-$ are copied from the online network every fixed number of environment steps; they are not directly optimized on the mini-batch.

### What Double DQN changes

Standard DQN lets the target network both select the maximum next-state Q-value and evaluate it. Double DQN uses the **online network to select** the next action and the **target network to evaluate** that selected action:

$$
a_j^* = \arg\max_{a'}Q(s'_j,a';\theta),
$$

$$
y_j^{\mathrm{DoubleDQN}} = r_j + \gamma(1-d_j)Q(s'_j,a_j^*;\theta^-).
$$

Double DQN uses the same replay memory, neural-network architecture, optimization procedure, and periodic target synchronization. Reducing overestimation does not guarantee a higher return in every run.

## Environment and implementation

- **Environment:** `CartPole-v1` (Gymnasium); four continuous observations and two discrete actions (left/right).
- **Agent network:** multilayer perceptron with `4 → 128 → 128 → 2` units, ReLU hidden layers, and linear Q-value outputs.
- **Framework:** PyTorch; Adam optimizer, uniform replay sampling, epsilon-greedy exploration, gradient clipping.
- **Illustrative settings in the notebook:** 400 training episodes; discount factor $\gamma=0.99$; mini-batch size 64; replay capacity 30,000; warm-up of 1,000 transitions; target synchronization every 250 environment steps; seed 42.
- **Evaluation:** greedy trained policy over 20 episodes compared with an independently sampled random-policy baseline.

These settings are configurable and are not claimed to be optimal. CartPole uses vector observations, so this implementation uses an MLP rather than an Atari-style convolutional network. The notebook also discusses the original pixel-based Atari DQN separately.

## Run the notebook

### Google Colab (recommended)

1. Open [`Deep_Q_Network_DQN_and_Double_DQN.ipynb`](Deep_Q_Network_DQN_and_Double_DQN.ipynb) in Google Colab.
2. Run the notebook from top to bottom. It includes a setup cell for Gymnasium and image/video support; Colab normally already provides NumPy, Pandas, Matplotlib, and PyTorch.
3. In the hyperparameter cell, set:

   ```python
   ALGORITHM = "dqn"         # standard DQN
   # ALGORITHM = "double_dqn"  # alternative: Double DQN
   ```

4. To compare the two methods, run the notebook separately for each algorithm. Each run reinitializes its networks and replay buffer. Double DQN output filenames receive a `double_dqn_` prefix, avoiding overwriting the DQN results.

### Local Jupyter environment

Use Python **3.10 or newer**. From the folder containing this README:

```bash
python -m venv .venv
# Activate the virtual environment using your platform's normal command.
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebook in JupyterLab and run its cells in order. Its first `%pip install` cell is also safe to run locally, although dependencies will already have been installed by `requirements.txt`.

## Repository structure

```text
09-Deep-Reinforcement-Learning/
├── README.md
├── requirements.txt
├── Deep_Q_Network_DQN_and_Double_DQN.ipynb
├── results/                    # figures, evaluation CSVs, and animation (after execution)
└── models/                     # trained PyTorch weights (after execution)
```

Illustrations are embedded in the notebook, so no external figure paths are required to read the chapter.

## Generated outputs

After training and evaluation, the notebook saves the following inside `results/`:

| File | Description |
|---|---|
| `reward_vs_episode.png` | Actual episode reward (unsmoothed) |
| `reward_with_moving_average.png` | Raw rewards plus 25-episode moving average |
| `training_loss.png` | Mini-batch training loss over optimizer steps |
| `epsilon_decay.png` | Exploration probability through training |
| `evaluation_comparison.png` | Mean return and standard deviation for random and trained policies |
| `evaluation_summary.csv` | Random-policy and trained-policy evaluation statistics |
| `selected_state_q_values.csv` | Predicted Q-values for selected observations |
| `trained_agent.gif` | Rendered greedy-agent episode |

The trained weights are saved to `models/dqn_cartpole.pt` or `models/double_dqn_cartpole.pt`, depending on the selected algorithm. For Double DQN, the `results/` names above are prefixed with `double_dqn_`.

**Note:** Generated graphs, animation, and weights are created when the notebook runs; they are not precomputed experimental results. Outcomes can vary with environment/version, hardware, and random seeds. A fair performance comparison requires equal training budgets and multiple seeds.

## References

- Mnih et al. (2013), [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602).
- Mnih et al. (2015), [*Human-level control through deep reinforcement learning*](https://doi.org/10.1038/nature14236).
- van Hasselt, Guez, and Silver (2016), [*Deep Reinforcement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461).
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition.
- [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and [PyTorch documentation](https://pytorch.org/docs/stable/index.html).
