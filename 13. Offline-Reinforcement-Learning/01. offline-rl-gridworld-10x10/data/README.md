# Dataset Description

## Main Training File

```text
gridworld_offline_dataset.csv
```

This file contains 42,986 fixed transitions from 1,000 episodes.

The learner uses the file as a static offline dataset.

## Minimum Offline RL Fields

```text
State_ID
Action_ID
Reward
Next_State_ID
Done
```

## Full Column Definitions

| Column | Meaning |
|---|---|
| Episode | Episode number |
| Step | Step within episode |
| Episode_Behavior_Type | Mixed Guided or Pure Random |
| State_Row | Current row |
| State_Col | Current column |
| State_ID | `row * 10 + col` |
| Action_ID | 0 Up, 1 Right, 2 Down, 3 Left |
| Action_Name | Human-readable action |
| Action_Source | Guided or Random logged action |
| Reward | 0, -5, or +10 |
| Next_State_Row | Next row |
| Next_State_Col | Next column |
| Next_State_ID | Encoded next state |
| Terminated | Goal reached |
| Truncated | 100-step limit reached |
| Done | Terminated OR Truncated |
| Hit_Blocked | Attempted blocked cell |
| Hit_Boundary | Attempted outside grid |
| Distance_To_Goal | Obstacle-aware distance before action |
| Next_Distance_To_Goal | Obstacle-aware distance after action |

## Supporting Files

- `episode_summary.csv`: one record per episode
- `grid_layout.csv`: the 10 × 10 map
- `blocked_cells.csv`: blocked coordinates
- `actions.csv`: action mapping

CSV was chosen because it can be read by the no-import trainer with built-in Python only.
