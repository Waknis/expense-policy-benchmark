import copy
import math
import random

import pytest

from expensebench.interactive_agent import (
    Actor, N_WEIGHTS, Reinforce, actor_inputs, add_gradient, comparison,
)
from expensebench.interactive_train import sample_episode


def observation():
    return {
        "policy": {"id": "p", "text": "test", "checks": [
            {"tool": "document", "field": "units", "operator": "at_most", "target": 2},
            {"tool": "ledger", "field": "channel", "operator": "equals", "target": "wire"},
        ]},
        "expense": {"id": "a", "amount_cents": 9500},
        "records": {},
        "remaining_queries": 3,
    }


def test_actor_reads_only_typed_policy_and_disclosed_values():
    obs = observation()
    actor = Actor.initialize(11)
    initial = actor.distribution(obs)
    alternate = copy.deepcopy(obs)
    alternate["expense"] = {"id": "different", "amount_cents": 1}
    alternate["policy"].update(id="different", text="unrelated prose")
    assert actor.distribution(alternate) == initial
    alternate["records"]["document"] = {"id": "x", "field": "units", "value": 2}
    alternate["remaining_queries"] -= 1
    state, legal = actor_inputs(alternate)
    assert state != initial[0]
    assert "lookup:document" not in dict(legal)
    check = alternate["policy"]["checks"][0]
    assert comparison(check, alternate["records"]) == "satisfies"
    alternate["records"]["document"]["value"] = 3
    assert comparison(check, alternate["records"]) == "violates"
    alternate["records"]["document"]["value"] = None
    assert comparison(check, alternate["records"]) == "missing"
    alternate["remaining_queries"] = 0
    assert len(actor_inputs(alternate)[1]) == 3


def test_gradient_matches_finite_difference_even_when_two_tools_share_one_role():
    obs = observation()
    actor = Actor.initialize(23)
    _, (state, roles, probabilities, choice) = actor.choose(obs, random.Random(7))
    gradient = [0.0] * N_WEIGHTS
    advantage, entropy_weight = 1.35, 0.03
    add_gradient(gradient, (state, roles, probabilities, choice), advantage, entropy_weight)

    def objective():
        _, _, p = actor.distribution(obs)
        return advantage * math.log(p[choice]) - entropy_weight * sum(x * math.log(x) for x in p)

    epsilon = 1e-5
    for i in range(N_WEIGHTS):
        weight = actor.weights[i]
        actor.weights[i] = weight + epsilon
        above = objective()
        actor.weights[i] = weight - epsilon
        below = objective()
        actor.weights[i] = weight
        assert gradient[i] == pytest.approx((above - below) / (2 * epsilon), abs=1e-8)


def test_scalar_reward_changes_policy_and_checkpoint_roundtrips(tmp_path):
    class OneStep:
        def reset(self):
            return observation()

        def step(self, action):
            reward = 2.0 if action == "review" else -1.0
            # Guard against training on private grader diagnostics.
            return None, reward, True, object()

    env = OneStep()
    actor = Actor.initialize(37)
    rng = random.Random(37)
    learner = Reinforce(actor)
    _, actions, initial = actor.distribution(observation())
    index = next(i for i, (action, _) in enumerate(actions) if action == "review")
    for _ in range(25):
        learner.update([sample_episode(actor, env, rng) for _ in range(32)])
    assert actor.distribution(observation())[2][index] > initial[index] + 0.25
    path = tmp_path / "actor.json"
    actor.save(path)
    assert Actor.load(path).weights == actor.weights
