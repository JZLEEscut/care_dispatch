"""Minimal event queue for the event-driven environment."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Event:
    time: float
    order: int
    event_type: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = 0

    def push(self, time: float, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self._counter += 1
        heapq.heappush(self._heap, Event(float(time), self._counter, event_type, payload or {}))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float | None:
        return self._heap[0].time if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
