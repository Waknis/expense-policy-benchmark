import pytest

gym = pytest.importorskip("gymnasium")
import numpy as np
from gymnasium.utils.env_checker import check_env

from expensebench.gym_env import GymExpenseEnv
from expensebench.interactive_agent import STATUSES, comparison
from expensebench.interactive_env import ACTIONS, TOOLS, ExpenseEnv, load_interactive_split


def test_gym_contract_and_seeded_case_selection():
    env = GymExpenseEnv(include_raw=True)
    check_env(env, skip_render_check=True)
    first, first_info = env.reset(seed=71)
    second, second_info = env.reset(seed=71)
    assert first_info["raw_observation"] == second_info["raw_observation"]
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert env.observation_space.contains(first)
    assert np.all(first["check_status"] == STATUSES.index("unread"))
    assert np.all(first["tools_read"] == 0)
    assert np.all(first_info["action_mask"] == 1)


def test_gym_features_and_step_rewards_match_core():
    cases, gold = load_interactive_split("train")
    case = cases[0]
    env = GymExpenseEnv()
    core = ExpenseEnv(case, gold[case["case_id"]])
    raw = core.reset()
    obs, info = env.reset(options={"case_id": case["case_id"]})
    assert "raw_observation" not in info
    checks = raw["policy"]["checks"]
    assert obs["check_sources"].tolist() == [TOOLS.index(check["tool"]) for check in checks]
    actions = [f"lookup:{check['tool']}" for check in checks] + ["approve"]
    for action in actions:
        raw, core_reward, core_done, core_info = core.step(action)
        obs, reward, done, truncated, info = env.step(ACTIONS.index(action))
        assert env.observation_space.contains(obs)
        assert (reward, done, truncated) == (core_reward, core_done, False)
        assert {k: v for k, v in info.items() if k != "action_mask"} == core_info
        assert obs["check_status"].tolist() == [
            STATUSES.index(comparison(check, raw["records"])) for check in checks
        ]
        if not done:
            assert info["action_mask"][ACTIONS.index(action)] == 0
    assert not info["action_mask"].any()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)


def test_budget_repeat_and_out_of_space_actions():
    env = GymExpenseEnv()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset(seed=5)
    with pytest.raises(ValueError, match="outside"):
        env.step(-1)
    env.step(0)
    _, reward, done, truncated, info = env.step(0)
    assert (reward, done, truncated, info["reason"]) == (-3.0, True, False, "repeated_lookup")
    env.reset(seed=5)
    for index in range(3):
        obs, _, done, _, info = env.step(index)
        assert not done
    assert obs["remaining_queries"] == 0
    assert info["action_mask"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    _, reward, done, truncated, info = env.step(3)
    assert (reward, done, truncated, info["reason"]) == (-3.0, True, False, "lookup_budget_exhausted")
    with pytest.raises(ValueError, match="Unknown case ID"):
        env.reset(options={"case_id": "absent"})
