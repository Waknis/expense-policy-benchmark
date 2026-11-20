"""Optional Gymnasium interface for the same interactive grader and lookup budget."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .interactive_agent import STATUSES, comparison
from .interactive_env import ACTIONS, MAX_QUERIES, TOOLS, ExpenseEnv, load_interactive_split


class GymExpenseEnv(gym.Env):
    """Sample an expense from one split or reset to a specific case ID.

    Action indices follow interactive_env.ACTIONS. The two ordered policy
    checks are represented by their source indices in TOOLS and comparisons
    in interactive_agent.STATUSES. Unread records stay unread until retrieved.
    Terminal actions use the core environment's default trace citations.
    """

    metadata = {"render_modes": []}

    def __init__(self, split="train", *, data_dir=None, include_raw=False):
        super().__init__()
        self._cases, self._gold = load_interactive_split(split, data_dir)
        self._by_id = {case["case_id"]: case for case in self._cases}
        self._episode = None
        self._include_raw = include_raw
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "check_sources": spaces.MultiDiscrete([len(TOOLS)] * 2),
            "check_status": spaces.MultiDiscrete([len(STATUSES)] * 2),
            "tools_read": spaces.MultiBinary(len(TOOLS)),
            "remaining_queries": spaces.Discrete(MAX_QUERIES + 1),
        })

    def _encode(self, visible, info, done):
        checks = visible["policy"]["checks"]
        records = visible["records"]
        remaining = visible["remaining_queries"]
        tools_read = np.array([source in records for source in TOOLS], dtype=np.int8)
        obs = {
            "check_sources": np.array([TOOLS.index(check["tool"]) for check in checks], dtype=np.int64),
            "check_status": np.array([STATUSES.index(comparison(check, records)) for check in checks],
                                     dtype=np.int64),
            "tools_read": tools_read,
            "remaining_queries": remaining,
        }
        mask = np.ones(len(ACTIONS), dtype=np.int8)
        mask[:len(TOOLS)] = (1 - tools_read) if remaining else 0
        if done:
            mask[:] = 0
        info = {**info, "action_mask": mask}
        if self._include_raw:
            info["raw_observation"] = visible
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        case_id = (options or {}).get("case_id")
        if case_id is None:
            case = self._cases[self.np_random.integers(len(self._cases))]
        else:
            if case_id not in self._by_id:
                raise ValueError(f"Unknown case ID in this split: {case_id}")
            case = self._by_id[case_id]
        self._episode = ExpenseEnv(case, self._gold[case["case_id"]])
        return self._encode(self._episode.reset(), {}, False)

    def step(self, action):
        if self._episode is None:
            raise gym.error.ResetNeeded("Call reset() before taking an action.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action index outside {self.action_space}: {action}")
        visible, reward, done, info = self._episode.step(ACTIONS[int(action)])
        obs, info = self._encode(visible, info, done)
        if done:
            self._episode = None
        return obs, reward, done, False, info
