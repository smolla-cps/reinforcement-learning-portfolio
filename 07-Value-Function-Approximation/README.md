# 07 — Value Function Approximation

A reinforcement learning portfolio notebook explaining **value function approximation (VFA)** from first principles, followed by implementations of **Approximate Q-Learning, SARSA, and Expected SARSA** in Gymnasium's `MountainCar-v0` environment.

The emphasis is on understanding how a model predicts values from state features, how temporal-difference (TD) targets are constructed, how gradient updates learn the weights, and how a parameterized model can replace a large Q-table.

## Learning objectives

- Understand why tabular value functions become difficult to use with large or continuous state spaces.
- Distinguish **environment states**, **feature vectors**, **model weights**, **predictions**, and **TD targets**.
- Explain linear and nonlinear value function approximation, including feature engineering.
- Derive the squared-error loss and the semi-gradient TD weight update.
- Calculate every step of an example: Q-value predictions, targets, TD error, loss, gradients, individual weight changes, and new predictions.
- See how learned weights can estimate values for states that do not have individual entries in a lookup table.
- Implement and compare Approximate Q-Learning, SARSA, and Expected SARSA in a continuous-state environment.

## Notebook organization

| Part | Content |
| --- | --- |
| **Foundations** | Tabular memory growth, parameterization, states and features, linear state-value and action-value functions. |
| **I — Derivations** | How weights are learned; prediction versus target; why the loss contains $\frac{1}{2}$; derivation of gradient and semi-gradient TD updates. |
| **II — Manual calculation** | Five-feature, three-action example with next-state predictions, Q-Learning target, TD error, loss, all individual weight updates, and an unseen-state prediction. |
| **III — TD control methods** | Q-Learning, SARSA, and Expected SARSA with the **same transition**, showing how different TD targets change the updates. |
| **IV — MountainCar experiment** | Radial basis function (RBF) features, training and evaluation, reward plots, update-norm convergence view, value surface, policy map, and GIF rollout. |

## Core equations

The notebook uses a linear action-value approximator with a separate weight vector for each discrete action:

$$
\hat Q(s,a;\mathbf W)=\mathbf w_a^T\boldsymbol\phi(s)
$$

For a transition $(S_t,A_t,R_{t+1},S_{t+1})$, define the error between the **target** $y_t$ and the **current prediction**:

$$
\delta_t=y_t-\hat Q(S_t,A_t;\mathbf W)
$$

Using the half-squared-error loss $L_t=\tfrac12\delta_t^2$ and treating the TD target as fixed when differentiating, the linear **semi-gradient** update is:

$$
\boxed{\mathbf w_{A_t}\leftarrow\mathbf w_{A_t}+\alpha\delta_t\boldsymbol\phi(S_t)}
$$

The three algorithms use different targets (for a nonterminal transition):

| Algorithm | TD target |
| --- | --- |
| Approximate Q-Learning | $R_{t+1}+\gamma\max_{a'}\hat Q(S_{t+1},a')$ |
| Approximate SARSA | $R_{t+1}+\gamma\hat Q(S_{t+1},A_{t+1})$ |
| Approximate Expected SARSA | $R_{t+1}+\gamma\sum_{a'}\pi(a'\mid S_{t+1})\hat Q(S_{t+1},a')$ |

For terminal transitions, the target is the immediate reward without a bootstrapped next-state term.

## Practical problem: MountainCar-v0

**Environment:** [Gymnasium MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

- **Observation:** two continuous values — car **position** and **velocity**.
- **Actions:** `0` (push left), `1` (no push), `2` (push right).
- **Reward:** `-1` per time step under the standard environment setup.
- **Goal:** learn a policy that reaches the goal in fewer steps.

The implementation constructs an **RBF feature representation** from a $9\times9$ grid of state-space centers, plus one bias feature (**82 features** in total). For three actions, the linear model maintains **246 weights**, rather than an independently stored Q-value for every continuous state-action combination.

All three algorithms share the same representation and training settings in the notebook. By default, each is trained for **2,500 episodes** and then evaluated with **100 greedy episodes**. Outcomes depend on random seed, hyperparameters, and training conditions; the comparison shows one experimental run, not a universal ranking.

## Getting started

### Option 1 — Google Colab

1. Upload `07_Value_Function_Approximation.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Use **Runtime → Run all** to execute the cells in order.
3. The notebook includes a `%pip` cell that installs `gymnasium[classic-control]` and `imageio`. NumPy, pandas, Matplotlib, and IPython are normally available in Colab.

### Option 2 — Run locally

Use Python **3.10 or newer** and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `07_Value_Function_Approximation.ipynb` and run the cells in order. The notebook's in-cell `%pip` command is safe to rerun, but the installation from `requirements.txt` already supplies the listed packages.

## Outputs

Running the notebook produces:

- A comparison of Q-table storage as the number of state bins increases.
- A plot showing how a parameter changes a function.
- Tables checking the worked example and individual weight updates.
- An RBF-center visualization.
- **Actual episode reward vs. episode** for all three approximate TD methods.
- A separate **moving-average reward** plot.
- An episode-level **weight-update norm** plot as a convergence diagnostic (not a proof of convergence).
- Greedy evaluation results: mean reward, reward standard deviation, average episode length, and success rate.
- A learned state-value surface and greedy action map.
- A greedy policy rollout saved as **`mountaincar_vfa.gif`** when the GIF cell is run.

Plots are displayed in the notebook; the GIF is generated in the current working directory. Training and evaluation can take some time depending on the runtime.

## Repository files

```text
07-Value-Function-Approximation/
├── README.md
├── requirements.txt
└── 07_Value_Function_Approximation.ipynb
```

The generated `mountaincar_vfa.gif` is optional if you want to include an example rollout in the repository.

## Main takeaway

A tabular agent stores values for individual state-action pairs. A VFA agent stores **learned parameters** and uses a feature transformation plus those parameters to estimate values for new states:

$$
\boxed{s\rightarrow\boldsymbol\phi(s)\rightarrow\hat Q(s,a;\mathbf W)\rightarrow\text{action selection}}
$$

The trade-off is that approximation can generalize and reduce storage requirements, but accuracy and learning stability depend on the features, parameters, and update method.
