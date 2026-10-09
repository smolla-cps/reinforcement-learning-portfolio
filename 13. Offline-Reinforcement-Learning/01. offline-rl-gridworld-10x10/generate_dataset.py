"""
Regenerate the 10 x 10 GridWorld fixed offline RL dataset.

Only Python standard-library modules are used.
"""

import random
import csv
from collections import deque

GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)

BLOCKED = {
    (1, 2), (2, 2), (3, 2), (4, 2),
    (4, 3), (4, 4), (4, 5), (4, 6),
    (2, 6), (3, 6), (5, 6), (6, 6),
    (7, 6), (7, 7), (7, 8),
    (2, 8), (3, 8), (4, 8), (5, 8),
}

ACTIONS = {
    0: ("Up", -1, 0),
    1: ("Right", 0, 1),
    2: ("Down", 1, 0),
    3: ("Left", 0, -1),
}

NUM_EPISODES = 1000
MAX_STEPS = 100
PURE_RANDOM_EPISODE_PROBABILITY = 0.20
GUIDED_ACTION_PROBABILITY = 0.65
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def inside(state):
    return 0 <= state[0] < 10 and 0 <= state[1] < 10

def environment_step(state, action_id):
    _, dr, dc = ACTIONS[action_id]
    candidate = (state[0] + dr, state[1] + dc)

    hit_boundary = not inside(candidate)
    hit_blocked = candidate in BLOCKED

    if hit_boundary or hit_blocked:
        return state, -5, False, hit_blocked, hit_boundary

    if candidate == GOAL:
        return candidate, 10, True, False, False

    return candidate, 0, False, False, False

def build_distance_map():
    distance = {GOAL: 0}
    queue = deque([GOAL])

    while queue:
        state = queue.popleft()

        for action_id in ACTIONS:
            _, dr, dc = ACTIONS[action_id]
            predecessor = (state[0] - dr, state[1] - dc)

            if (
                inside(predecessor)
                and predecessor not in BLOCKED
                and predecessor not in distance
            ):
                distance[predecessor] = distance[state] + 1
                queue.append(predecessor)

    return distance

DISTANCE = build_distance_map()

def good_actions(state):
    current = DISTANCE.get(state, 999)
    result = []

    for action_id in ACTIONS:
        next_state, _, _, hit_blocked, hit_boundary = environment_step(
            state,
            action_id,
        )

        if hit_blocked or hit_boundary:
            continue

        if DISTANCE.get(next_state, 999) < current:
            result.append(action_id)

    return result

headers = [
    "Episode",
    "Step",
    "Episode_Behavior_Type",
    "State_Row",
    "State_Col",
    "State_ID",
    "Action_ID",
    "Action_Name",
    "Action_Source",
    "Reward",
    "Next_State_Row",
    "Next_State_Col",
    "Next_State_ID",
    "Terminated",
    "Truncated",
    "Done",
    "Hit_Blocked",
    "Hit_Boundary",
    "Distance_To_Goal",
    "Next_Distance_To_Goal",
]

with open(
    "gridworld_offline_dataset.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)
    writer.writerow(headers)

    for episode in range(1, NUM_EPISODES + 1):

        pure_random = (
            random.random()
            <
            PURE_RANDOM_EPISODE_PROBABILITY
        )

        episode_type = (
            "Pure Random"
            if pure_random
            else "Mixed Guided"
        )

        state = START

        for step_number in range(MAX_STEPS):

            if pure_random:
                action_id = random.randrange(4)
                action_source = "Random"

            else:
                candidates = good_actions(state)

                if (
                    candidates
                    and
                    random.random()
                    <
                    GUIDED_ACTION_PROBABILITY
                ):
                    action_id = random.choice(candidates)
                    action_source = "Guided"

                else:
                    action_id = random.randrange(4)
                    action_source = "Random"

            (
                next_state,
                reward,
                terminated,
                hit_blocked,
                hit_boundary,
            ) = environment_step(
                state,
                action_id,
            )

            truncated = (
                step_number
                ==
                MAX_STEPS - 1
                and
                not terminated
            )

            done = terminated or truncated

            writer.writerow([
                episode,
                step_number,
                episode_type,
                state[0],
                state[1],
                state[0] * 10 + state[1],
                action_id,
                ACTIONS[action_id][0],
                action_source,
                reward,
                next_state[0],
                next_state[1],
                next_state[0] * 10 + next_state[1],
                int(terminated),
                int(truncated),
                int(done),
                int(hit_blocked),
                int(hit_boundary),
                DISTANCE.get(state, -1),
                DISTANCE.get(next_state, -1),
            ])

            state = next_state

            if done:
                break

print("gridworld_offline_dataset.csv generated.")
