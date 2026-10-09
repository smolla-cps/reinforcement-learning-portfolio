# Data

This repository does not store the full PointMaze dataset in GitHub.

The corrected notebook downloads the dataset through Minari:

```python
DATASET_ID = "D4RL/pointmaze/medium-v2"

dataset = minari.load_dataset(
    DATASET_ID,
    download=True,
)
```

## Dataset Meaning

The dataset contains fixed goal-conditioned navigation trajectories from PointMaze Medium.

A 2-DoF point mass moves through a maze under continuous x/y forces.

The notebook states that the dataset contains:

```text
1,000,000 fixed transitions
4,752 stored episodes
```

with waypoint-based behavior and added action noise.

## Observation Structure

The Minari/Gymnasium observation is a dictionary:

```text
observation
achieved_goal
desired_goal
```

The notebook flattens these in the fixed order:

```text
observation → achieved_goal → desired_goal
```

to produce an 8-dimensional numerical state.

## Action Structure

The action contains two continuous values:

```text
force in x direction
force in y direction
```

## Training Transition Representation

The notebook constructs:

```text
observations
actions
rewards
next_observations
terminations
truncations
```

and later creates a Q-learning-style dictionary:

```text
observations
actions
rewards
next_observations
terminals
timeouts
```

Large downloaded dataset files should not be committed to GitHub.
