# Reward Design in Reinforcement Learning

This folder focuses on one of the most important parts of reinforcement learning: **designing reward functions that lead an agent toward the behavior we actually want**.

A reinforcement-learning algorithm does not directly understand concepts such as *reach the goal quickly*, *stay stable*, *avoid unnecessary motion*, or *land safely*. It learns from the numerical reward signal provided by the environment. Because of this, a poorly designed reward can make learning extremely slow or can lead the agent to discover behavior that maximizes reward without solving the intended task.

The notebooks in this folder build the topic in three stages:

$$
\boxed{
\text{Reward Types and Criteria}
\rightarrow
\text{How RL Agents Learn}
\rightarrow
\text{Designing Effective Rewards}
}
$$

---

## Folder Contents

| Notebook | Main Focus |
|---|---|
| [`01_reward_types_and_criteria.ipynb`](01_reward_types_and_criteria.ipynb) | Reward types, reward criteria, design requirements, and practical sanity checks |
| [`02_how do rl agent really learn.ipynb`](02_how%20do%20rl%20agent%20really%20learn.ipynb) | How Dynamic Programming, Monte Carlo, TD, SARSA, and Q-Learning use reward information to learn |
| [`03_designing_effective_reward_functions.ipynb`](03_designing_effective_reward_functions.ipynb) | Step-by-step reward design using GridWorld, CartPole, Mountain Car, and Lunar Lander |

---

# 1. Reward Types and Reward Criteria

The first notebook separates two ideas that are easy to confuse:

$$
\boxed{
\text{Reward Function}
\neq
\text{Reward Criterion}
}
$$

The **reward function** determines the immediate reward after a transition:

$$
R_{t+1}=R(S_t,A_t,S_{t+1})
$$

The **reward criterion** determines how rewards are accumulated over time to form the objective that the agent tries to maximize.

The notebook studies several common reward structures:

- sparse or terminal rewards,
- dense or progress rewards,
- step penalties or time-cost rewards,
- shaped rewards,
- composite rewards,
- extrinsic and intrinsic rewards.

It then compares three major reward criteria.

### Total episodic reward

$$
G=\sum_{t=1}^{T}R_t
$$

### Discounted return

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
$$

### Long-run average reward

$$
\rho=
\lim_{n\rightarrow\infty}
\frac{1}{n}
\mathbb{E}
\left[
\sum_{t=1}^{n}R_t
\right]
$$

The notebook also develops practical criteria for evaluating a reward function. A useful reward should be aligned with the real objective, informative enough for learning, correctly scaled, consistent with the desired time preference, difficult to exploit, compatible with constraints, and validated by observing the learned behavior.

---

# 2. How Reinforcement Learning Agents Actually Learn

Reward design becomes easier to understand once we know **how the learning algorithm uses the reward signal**.

The second notebook follows the progression

$$
\boxed{
\text{MDP}
\rightarrow
\text{Dynamic Programming}
\rightarrow
\text{Monte Carlo}
\rightarrow
\text{Temporal Difference}
\rightarrow
\text{SARSA}
\rightarrow
\text{Q-Learning}
}
$$

It begins with Markov Decision Processes, the environment model, and exploration versus exploitation. It then explains how value functions and action-value functions are updated from reward information.

### Monte Carlo

Monte Carlo methods wait for a complete return before updating:

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
G_t-V(S_t)
\right]
$$

### Temporal Difference

TD methods update before the episode is finished by bootstrapping from the next estimate:

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
\right]
$$

### SARSA

SARSA learns from the action actually selected by the current policy:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
-
Q(S_t,A_t)
\right]
$$

### Q-Learning

Q-Learning uses the best estimated next action:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma\max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right]
$$

These methods differ in how they construct their learning targets, but the reward remains a central part of every update. If the reward signal encourages the wrong behavior, the learning algorithm can correctly optimize the wrong objective.

---

# 3. Designing Effective Reward Functions

The third notebook applies the previous ideas to increasingly complex environments.

The central lesson is:

$$
\boxed{
\text{The agent learns what the reward function asks for, not what we intended}
}
$$

## GridWorld

GridWorld is used to compare three simple reward designs.

### Zero reward until the goal

Giving

$$
R=0
$$

for ordinary steps and a positive reward only at the goal creates a sparse signal. The agent receives little information about whether an intermediate action moved it closer to or farther from success.

### Positive reward for every step

Giving positive reward for movement can produce the wrong incentive because the agent may increase its total reward by taking more steps instead of reaching the goal quickly.

### Negative reward for every step

A step cost such as

$$
R=-1
$$

makes unnecessary movement expensive. Reaching the goal in fewer steps therefore produces a better return.

This simple example shows that changing only the sign and timing of reward can completely change what behavior is optimal.

---

## Sparse Rewards vs. Shaped Rewards

A **sparse reward** provides useful feedback only occasionally. A **shaped reward** adds intermediate information that indicates whether the state or transition is improving.

A common difference-based shaping idea is

$$
\boxed{
R_t=\Phi_t-\Phi_{t-1}
}
$$

where $\Phi_t$ is a shaping score describing the quality of the current state.

If

$$
\Phi_t>\Phi_{t-1},
$$

the state improved and the shaping reward is positive.

If

$$
\Phi_t<\Phi_{t-1},
$$

the state became worse and the shaping reward is negative.

Reward shaping can make learning faster, but it must be designed carefully because every added term creates a new incentive.

---

# 4. CartPole Reward Shaping

For CartPole, simply rewarding survival does not explain *how stable* the current state is.

The notebook develops a shaping score using variables such as:

- pole angle,
- cart position,
- cart velocity,
- pole angular velocity.

A representative shaping form is

$$
\boxed{
\Phi_t=
-\sqrt{
pa_t^2+cp_t^2+cv_t^2+pav_t^2
}
}
$$

The best state has a shaping score close to zero. Larger deviations make the score more negative.

The reward can then be based on improvement:

$$
\boxed{
R_t=\Phi_t-\Phi_{t-1}
}
$$

The notebook also explains why raw signed errors can be misleading. If deviations in both directions are equally undesirable, the reward should depend on the **magnitude** of the error. Squared or absolute-value terms can therefore be useful:

$$
(+4)^2=(-4)^2
$$

and

$$
|+4|=|-4|.
$$

---

# 5. Mountain Car and Reward Hacking

Mountain Car demonstrates an important lesson: **a reward can make learning faster and still produce undesirable behavior**.

An initial shaped reward gives feedback based on progress toward the goal and velocity. This helps the agent learn faster, but the agent discovers that repeatedly oscillating can collect additional shaping reward before finally reaching the goal.

The agent is not violating the reward function. It is optimizing it.

This is an example of an unintended incentive or **reward hacking**.

The improved reward therefore changes the objective so that:

- ordinary steps remain costly,
- being closer to the goal is better,
- useful velocity reduces the penalty,
- the large positive reward is reserved for actually reaching the goal.

The example demonstrates an important validation rule:

$$
\boxed{
\text{Always inspect learned behavior, not only the reward curve}
}
$$

---

# 6. Lunar Lander: Combining Multiple Objectives

Lunar Lander shows how a reward function can combine several behavioral objectives.

The state contains eight components representing:

$$
[x,\ y,\ v_x,\ v_y,\ \theta,\ \dot{\theta},\ c_L,\ c_R]
$$

where the terms describe position, velocity, orientation, angular velocity, and left/right leg contact.

The action space contains four discrete actions:

1. do nothing,
2. fire the left orientation engine,
3. fire the main engine,
4. fire the right orientation engine.

The reward combines several considerations, including:

- movement toward a stable landing configuration,
- successful leg contact,
- successful landing,
- penalties for unnecessary engine use.

The shaping reward again uses the change in a state-quality score:

$$
R_t=\Phi_t-\Phi_{t-1}.
$$

The Lunar Lander example demonstrates how one reward function can balance position, velocity, orientation, contact, success, and control effort.

---

# Practical Reward-Design Workflow

A useful reward-design process is:

1. **Define the true objective.** Write down exactly what successful behavior should look like.
2. **Identify undesirable shortcuts.** Ask how an agent could receive high reward without doing what you intended.
3. **Choose sparse or shaped feedback.** Use additional shaping only when the original signal does not provide enough learning information.
4. **Check the time incentive.** Decide whether shorter, longer, or continuing behavior should be preferred.
5. **Check sign and scale.** Make sure positive and negative terms have the intended relative importance.
6. **Test simple transitions manually.** Calculate rewards for good, bad, improving, and worsening transitions before training.
7. **Train and inspect behavior.** Do not judge a reward function from total reward alone.
8. **Revise the reward when behavior exposes a loophole.**

---

# Key Lessons

Reward design and learning algorithm design are closely connected.

A sparse reward may correctly describe the task but provide too little learning signal. A dense reward may accelerate learning but accidentally reward behavior that is unrelated to the real objective. Step penalties can encourage efficiency, while poor positive shaping can encourage delay. Composite rewards can express several objectives, but their weights determine the trade-offs the agent actually learns.

The most important principle is:

$$
\boxed{
\text{A good reward function makes desirable behavior the easiest way to obtain high return}
}
$$

Before trusting a trained policy, examine both its numerical performance and its actual behavior.

---

# Tools Used

The notebooks use:

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook / Google Colab

No specialized reinforcement-learning library is required for the core demonstrations.

---

# Recommended Study Order

Start with `01_reward_types_and_criteria.ipynb` to understand the vocabulary and mathematical criteria used to describe rewards.

Continue with `02_how do rl agent really learn.ipynb` to see how rewards enter Dynamic Programming, Monte Carlo, TD, SARSA, and Q-Learning updates.

Finish with `03_designing_effective_reward_functions.ipynb` to apply those ideas to practical reward-design problems and study both successful shaping and reward-design failures.

---

## Repository Context

This folder is part of a larger reinforcement-learning portfolio covering foundational RL concepts, value-based methods, policy-based methods, deep reinforcement learning, and practical environment design.
