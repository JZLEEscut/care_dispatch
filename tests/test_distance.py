from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heuristics.cost import distance


def test_distance_symmetric_and_zero():
    a = (1.0, 2.0)
    b = (4.0, 6.0)
    assert distance(a, b) == distance(b, a)
    assert distance(a, a) == 0.0
