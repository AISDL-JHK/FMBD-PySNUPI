"""Observer lifecycle protocol."""

from __future__ import annotations

from typing import Any, Protocol


class Observer(Protocol):
    def on_start(self, simulation: Any, context: Any) -> None: ...
    def on_step(self, simulation: Any, result: Any) -> None: ...
    def on_finish(self, simulation: Any) -> None: ...
