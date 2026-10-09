# Data

The raw Minari dataset is downloaded automatically and is not committed to this repository.

Dataset ID:

```text
D4RL/minigrid/fourrooms-random-v0
```

The notebook reads the data directly through Minari.

For deep offline-RL methods, the observation dictionary is converted to a numeric vector:

```text
image:     7 × 7 × 3 = 147 values
direction:               1 value
----------------------------------
numeric state:          148 values
```

The mission text is excluded from the numeric state because it is constant for this task.

Generated large data files should remain outside version control.
