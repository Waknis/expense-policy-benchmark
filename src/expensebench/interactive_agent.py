"""A small on-policy actor for the typed, partially observed expense task."""

import json
import math
from pathlib import Path
import random

from .interactive_env import TOOLS


DECISIONS = ("approve", "review", "reject")
ROLES = ("first_source", "second_source", "other_source", *DECISIONS)
# States are pairs of public comparisons, in the order given in the policy.
STATUSES = ("unread", "missing", "satisfies", "violates")
N_STATES = len(STATUSES) ** 2
N_WEIGHTS = (N_STATES + 1) * len(ROLES)
LEARNING_RATE = 0.025
ENTROPY_WEIGHT = 0.025


def comparison(check, records):
    """Compute a named predicate using only evidence already returned by a tool."""
    source = check["tool"]
    if source not in records:
        return "unread"
    record = records[source]
    if record["field"] != check["field"]:
        raise ValueError("Fetched field differs from the public policy predicate.")
    value, target = record["value"], check["target"]
    if value is None:
        return "missing"
    operator = check["operator"]
    if operator == "equals":
        satisfies = value == target
    elif operator == "at_most":
        satisfies = value <= target
    elif operator == "at_least":
        satisfies = value >= target
    else:
        raise ValueError(f"Unsupported public predicate: {operator}")
    return "satisfies" if satisfies else "violates"


def actor_inputs(observation):
    """Map visible inputs to a state index and available (action, shared role) pairs."""
    first, second = observation["policy"]["checks"]
    records = observation["records"]
    first_state = STATUSES.index(comparison(first, records))
    second_state = STATUSES.index(comparison(second, records))
    state = first_state * len(STATUSES) + second_state
    actions = []
    if observation["remaining_queries"] > 0:
        for tool in TOOLS:
            if tool in records:
                continue
            role = 0 if tool == first["tool"] else 1 if tool == second["tool"] else 2
            actions.append((f"lookup:{tool}", role))
    actions.extend((decision, 3 + i) for i, decision in enumerate(DECISIONS))
    return state, actions


def categorical(logits):
    peak = max(logits)
    numerators = [math.exp(value - peak) for value in logits]
    total = sum(numerators)
    return [value / total for value in numerators]


class Actor:
    """Shared role biases plus role logits for each observed comparison pair."""

    def __init__(self, weights):
        if len(weights) != N_WEIGHTS or not all(math.isfinite(x) for x in weights):
            raise ValueError("Invalid actor weights.")
        self.weights = list(weights)

    @classmethod
    def initialize(cls, seed):
        rng = random.Random(seed)
        return cls([rng.gauss(0.0, 0.05) for _ in range(N_WEIGHTS)])

    def distribution(self, observation):
        state, actions = actor_inputs(observation)
        offset = (state + 1) * len(ROLES)
        logits = [self.weights[role] + self.weights[offset + role] for _, role in actions]
        return state, actions, categorical(logits)

    def choose(self, observation, rng=None):
        state, actions, probabilities = self.distribution(observation)
        if rng is None:
            choice = max(range(len(actions)), key=probabilities.__getitem__)
        else:
            draw = rng.random()
            cumulative = 0.0
            choice = len(actions) - 1
            for i, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    choice = i
                    break
        roles = [role for _, role in actions]
        return actions[choice][0], (state, roles, probabilities, choice)

    def save(self, path):
        data = {"format": 1, "roles": list(ROLES), "statuses": list(STATUSES), "weights": self.weights}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("format"), data.get("roles"), data.get("statuses")) != (1, list(ROLES), list(STATUSES)):
            raise ValueError("Checkpoint feature schema does not match this actor.")
        return cls(data["weights"])


def add_gradient(gradient, decision, advantage, entropy_weight):
    """Accumulate advantage * grad(log pi) + beta * grad(entropy) for one choice."""
    state, roles, probabilities, choice = decision
    offset = (state + 1) * len(ROLES)
    mean_log = sum(p * math.log(p) for p in probabilities if p)
    for i, (role, probability) in enumerate(zip(roles, probabilities, strict=True)):
        change = advantage * (float(i == choice) - probability)
        if probability:
            change -= entropy_weight * probability * (math.log(probability) - mean_log)
        gradient[role] += change
        gradient[offset + role] += change


class Reinforce:
    """Monte Carlo policy gradient with a state baseline from previous batches."""

    def __init__(self, actor):
        self.actor = actor
        self.baselines = [0.0] * N_STATES
        self.first_moment = [0.0] * N_WEIGHTS
        self.second_moment = [0.0] * N_WEIGHTS
        self.updates = 0

    def update(self, episodes):
        if not episodes:
            raise ValueError("Need at least one episode.")
        gradient = [0.0] * N_WEIGHTS
        returns = [0.0] * N_STATES
        visits = [0] * N_STATES
        for episode in episodes:
            future = 0.0
            for decision, reward in reversed(episode):
                future += reward
                state = decision[0]
                add_gradient(gradient, decision, future - self.baselines[state], ENTROPY_WEIGHT)
                returns[state] += future
                visits[state] += 1
        for state, n in enumerate(visits):
            if n:
                self.baselines[state] += 0.1 * (returns[state] / n - self.baselines[state])
        self.updates += 1
        for i, total in enumerate(gradient):
            value = total / len(episodes)
            self.first_moment[i] = 0.9 * self.first_moment[i] + 0.1 * value
            self.second_moment[i] = 0.999 * self.second_moment[i] + 0.001 * value * value
            m = self.first_moment[i] / (1 - 0.9 ** self.updates)
            v = self.second_moment[i] / (1 - 0.999 ** self.updates)
            self.actor.weights[i] += LEARNING_RATE * m / (math.sqrt(v) + 1e-8)
