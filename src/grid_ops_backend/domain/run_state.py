from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class RunState:
    run_id: str
    network_id: str
    seed: int
    events: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def with_event(self, event_type: str, payload: dict[str, object]) -> "RunState":
        new_event = {"type": event_type, "payload": payload}
        return replace(self, events=self.events + (new_event,))
