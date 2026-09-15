# Model-Free Tabular Reinforcement Learning

This folder demonstrates how values and policies can be learned directly from sampled experience without being given transition probabilities.

## Project Structure

```text
03-model-free-tabular-rl/
│
├── README.md
├── requirements.txt
├── 01_blackjack_monte_carlo_vs_td.ipynb
├── 02_cliffwalking_sarsa_vs_q_learning.ipynb
└── 03_model_free_tabular_comparison.ipynb
```

## 1. Blackjack: Monte Carlo Prediction vs. TD(0)

The Blackjack notebook compares two model-free prediction methods under the same fixed policy.

- **State:** `(player sum, dealer showing, usable ace)`
- **Actions:** Hit or Stick
- **Transition:** sampled card outcomes; transition probabilities are not supplied
- **Reward:** win `+1`, draw `0`, loss `-1`
- **Learned quantity:** $V^\pi(s)$

Monte Carlo uses the complete sampled return:

$$V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]$$

TD(0) bootstraps after each transition:

$$V(S_t)\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$$

The notebook prints a complete episode, every return calculation, every first-visit Monte Carlo update, a complete TD(0) update trace, learning curves, value heatmaps, and simulations showing the learned value estimates along the policy trajectory.

## 2. CliffWalking: SARSA vs. Q-Learning

The CliffWalking notebook compares on-policy and off-policy TD control.

- **State:** grid position
- **Actions:** Up, Right, Down, Left
- **Transition:** one grid movement; cliff returns the agent to the start
- **Reward:** normal step `-1`, cliff `-100`
- **Learned quantity:** $Q(s,a)$

SARSA uses the actual next behavior action:

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$$

Q-Learning uses the greedy next-state target:

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)]$$

The notebook prints all values in representative updates, traces complete learning episodes step by step, plots return and cliff-fall learning curves, extracts the learned policies, and simulates both greedy policies one step at a time.

## 3. Model-Free Tabular Comparison

The comparison notebook connects the four methods:

| Method | Learns | Target | Role |
|---|---|---|---|
| Monte Carlo | $V(s)$ | $G_t$ | Prediction |
| TD(0) | $V(s)$ | $R+\gamma V(s')$ | Prediction |
| SARSA | $Q(s,a)$ | $R+\gamma Q(s',a')$ | On-policy control |
| Q-Learning | $Q(s,a)$ | $R+\gamma\max_aQ(s',a)$ | Off-policy control |

It also prints a numerical update for each method, compares their targets on a short sampled trajectory, and shows how tabular storage grows with the state space.

## Learning Progression

```text
Monte Carlo
    ↓
TD(0)
    ↓
SARSA
    ↓
Q-Learning
    ↓
Deep value methods
```

## Running the Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

