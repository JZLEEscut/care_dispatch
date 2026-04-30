"""Solomon VRPTW text-file loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SolomonCustomer:
    customer_id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


def load_solomon_file(path: str | Path) -> list[SolomonCustomer]:
    """Load customer rows from a Solomon VRPTW instance.

    Expected numeric row format:
    customer_id, x, y, demand, ready_time, due_time, service_time.
    Non-data header lines are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Solomon file not found: {path}")

    rows: list[SolomonCustomer] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(p) for p in parts[:7]]
        except ValueError:
            continue
        customer_id = int(vals[0])
        rows.append(
            SolomonCustomer(
                customer_id=customer_id,
                x=vals[1],
                y=vals[2],
                demand=vals[3],
                ready_time=vals[4],
                due_time=vals[5],
                service_time=vals[6],
            )
        )

    if not rows:
        raise ValueError(f"No Solomon customer rows found in {path}")
    return rows
