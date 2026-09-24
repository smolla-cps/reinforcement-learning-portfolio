# Monte Carlo Methods for Reinforcement Learning

This repository presents Monte Carlo methods for reinforcement learning through detailed mathematical examples and a custom **7×7 GridWorld** implementation.

The notebooks are organized to connect the theory with the algorithm step by step. Episodes are written explicitly, rewards are shown for each transition, returns are calculated backward, and the corresponding value or action-value updates are shown before the Python implementation.

## Notebooks

### 1. `Monte_Carlo_Methods_RL.ipynb`

This notebook develops the main Monte Carlo ideas and algorithms.

Topics include:

- Monte Carlo learning and complete returns
- Monte Carlo vs. Dynamic Programming
- model-free and episodic learning
- online vs. offline learning
- on-policy vs. off-policy learning
- First-Visit Monte Carlo Prediction
- Every-Visit Monte Carlo Prediction
- First-Visit vs. Every-Visit comparison
- Monte Carlo Control with Exploring Starts
- on-policy Monte Carlo Control with an $\epsilon$-greedy policy
- transition from Monte Carlo methods to Temporal-Difference learning

The return is calculated backward using

$$
G \leftarrow \gamma G + R_{t+1}.
$$

For prediction, the notebooks estimate

$$
V(s) \approx v_\pi(s).
$$

For control, action values are estimated using

$$
Q(s,a),
$$

and the policy is improved from those action-value estimates.

[Open the Monte Carlo Methods notebook](Monte_Carlo_Methods_RL.ipynb)

---

### 2. `Monte_Carlo_7x7_GridWorld(1).ipynb`

This notebook applies Monte Carlo methods to a custom **7×7 GridWorld**.

The environment includes:

- a fixed start state
- a goal state
- blocked cells
- four actions: up, down, left, and right
- step penalties
- a positive terminal reward

The notebook applies:

1. **First-Visit Monte Carlo Prediction**
2. **Every-Visit Monte Carlo Prediction**
3. **On-Policy First-Visit Monte Carlo Control with $\epsilon$-Greedy Exploration**

For each algorithm, a complete episode is shown and processed step by step.

For example, the return is updated backward as

$$
G \leftarrow \gamma G + R_{t+1}.
$$

For First-Visit MC, only the first occurrence of a state in the original forward episode is used.

For Every-Visit MC, every occurrence is used.

For MC Control, the algorithm estimates

$$
Q(s,a)
$$

and applies the sample-average update

$$
Q(s,a)
\leftarrow
Q(s,a)
+
\frac{1}{N(s,a)}
\left[
G-Q(s,a)
\right].
$$

The $\epsilon$-greedy policy is

$$
\pi(a\mid s)=
\begin{cases}
1-\epsilon+\dfrac{\epsilon}{|\mathcal A|},
& a=\arg\max_{a'}Q(s,a'),\\[6pt]
\dfrac{\epsilon}{|\mathcal A|},
& \text{otherwise}.
\end{cases}
$$

After training, the notebook shows:

- the learned greedy policy
- state-value estimates
- learning curves
- episode return
- episode length
- goal-reaching rate
- a step-by-step simulation using the learned policy
- the final path from the start state to the goal

[Open the 7×7 GridWorld notebook](Monte_Carlo_7x7_GridWorld%281%29.ipynb)

## Main Algorithms

### First-Visit Monte Carlo Prediction

For each episode:

1. Generate a complete episode.
2. Initialize

   $$
   G=0.
   $$

3. Process the episode backward:

   $$
   G\leftarrow\gamma G+R_{t+1}.
   $$

4. Update a state only if it is its first occurrence in the original forward episode.
5. Estimate

   $$
   V(s)=\operatorname{average}(\text{Returns}(s)).
   $$

### Every-Visit Monte Carlo Prediction

Every-Visit MC uses the same return calculation,

$$
G\leftarrow\gamma G+R_{t+1},
$$

but every occurrence of a state contributes to the estimate.

### Monte Carlo Control

Monte Carlo control learns action values:

$$
Q(s,a).
$$

The learned policy is obtained from

$$
\pi(s)=\arg\max_a Q(s,a).
$$

An $\epsilon$-greedy policy keeps exploration during training while favoring actions with larger estimated action values.

## Repository Structure

```text
.
├── Monte_Carlo_Methods_RL.ipynb
├── Monte_Carlo_7x7_GridWorld(1).ipynb
├── README.md
└── requirements.txt
```

## Requirements

The notebooks use:

- Python 3
- NumPy
- pandas
- Matplotlib

The GridWorld is implemented directly in Python, so no external reinforcement-learning environment package is required.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

### Google Colab

Upload either notebook to Google Colab and run the cells from top to bottom.

### Jupyter Notebook or JupyterLab

Clone or download the repository, install the requirements, and open the notebooks in your preferred Jupyter environment.

```bash
pip install -r requirements.txt
```

Then run the notebook cells in order.

## Learning Progression

The notebooks follow this progression:

$$
\boxed{
\text{Complete Episodes}
\rightarrow
\text{Returns}
\rightarrow
\text{MC Prediction}
\rightarrow
\text{Action Values}
\rightarrow
\text{MC Control}
\rightarrow
\text{Learned Policy}
}
$$

The next natural step after Monte Carlo methods is **Temporal-Difference learning**, where value estimates can be updated before an episode is complete.
