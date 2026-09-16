"""Per-run state shared through MAF invocation context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

RUN_STATE_KEY = "_agent_learning_run"


@dataclass(frozen=True)
class AgentFrameworkRunState:
    messages: Sequence[Any]


__all__ = ["RUN_STATE_KEY", "AgentFrameworkRunState"]
