from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pure_heuristic_policy import PureHeuristicPolicy
from src.baselines.rolling_ortools_policy import RollingORToolsPolicy
from src.baselines.rolling_alns_policy import RollingALNSPolicy
from src.env.dispatch_env import CareDispatchEnv
from src.scripts._script_utils import load_yaml


def test_benchmark_policies_return_valid_actions():
    cfg = load_yaml('configs/env/rc101_default.yaml')
    cfg['data']['num_tasks'] = 10
    cfg['data']['num_workers'] = 3
    env = CareDispatchEnv(cfg)
    state = env.reset(seed=0)
    for policy in [PureHeuristicPolicy({}), RollingORToolsPolicy({'rolling_ortools': {'time_limit_seconds': 0.1}}), RollingALNSPolicy({'rolling_alns': {'iterations': 3, 'time_limit_seconds': 0.1}})]:
        action = policy.select_action(state, env, deterministic=True)
        assert isinstance(action, int)
        assert 0 <= action <= len(env.workers)
