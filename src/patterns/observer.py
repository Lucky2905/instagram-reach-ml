"""
src/patterns/observer.py — Observer Pattern
============================================
Training lifecycle events are published by TrainingSubject and consumed
by zero-coupled observers. Models NEVER call observers directly.

Events published:
    TRAINING_START     — pipeline begins
    FOLD_COMPLETE      — CV fold finished (fired by CrossValidationDecorator)
    TRAINING_COMPLETE  — both models trained; metrics attached
    ERROR              — unexpected failure

Observers available:
    ConsoleObserver      — formatted stdout logging
    FileMetricsObserver  — JSON file persistence (appends across runs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
import json
import logging
import time

logger = logging.getLogger(__name__)


# ── Event value object ────────────────────────────────────────────────────────

class TrainingEvent:
    """Immutable snapshot of a training lifecycle moment."""

    __slots__ = ("event_type", "data", "timestamp")

    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"TrainingEvent(type={self.event_type!r}, data={self.data})"


# ── Abstract Observer ─────────────────────────────────────────────────────────

class TrainingObserver(ABC):
    """
    Abstract observer. Implement update() to react to training events.
    Observers are never referenced by the model — only by TrainingSubject.
    """

    @abstractmethod
    def update(self, event: TrainingEvent) -> None:
        """Handle an incoming training event."""

    @property
    @abstractmethod
    def observer_name(self) -> str:
        """Unique observer identifier for logging."""


# ── Concrete Observers ────────────────────────────────────────────────────────

class ConsoleObserver(TrainingObserver):
    """Prints richly-formatted training events to stdout."""

    _ICONS: Dict[str, str] = {
        "TRAINING_START":    "[START]",
        "FOLD_COMPLETE":     "[FOLD] ",
        "TRAINING_COMPLETE": "[DONE] ",
        "ERROR":             "[ERROR]",
    }

    def update(self, event: TrainingEvent) -> None:
        icon = self._ICONS.get(event.event_type, "[INFO] ")
        if event.event_type == "TRAINING_COMPLETE":
            border = "=" * 60
            logger.info("\n%s\n%s  %s\n%s\n%s", border, icon, event.event_type,
                        event.data, border)
        elif event.event_type == "ERROR":
            logger.error("%s %s: %s", icon, event.event_type, event.data)
        else:
            logger.info("%s  %s: %s", icon, event.event_type, event.data)

    @property
    def observer_name(self) -> str:
        return "ConsoleObserver"


class FileMetricsObserver(TrainingObserver):
    """
    Persists every training event to a JSON file.
    Appends to existing records, enabling multi-run trend analysis.
    """

    def __init__(self, output_path: Path):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = self._load_existing()

    def _load_existing(self) -> List[Dict]:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def update(self, event: TrainingEvent) -> None:
        record: Dict[str, Any] = {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
        }
        # Safely serialise all data values
        for k, v in event.data.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v

        self._records.append(record)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._records, f, indent=2, default=str)
        except OSError as exc:
            logger.error("[FileMetricsObserver] Write failed: %s", exc)

    @property
    def observer_name(self) -> str:
        return "FileMetricsObserver"

    @property
    def records(self) -> List[Dict]:
        return list(self._records)


# ── Subject (Publisher) ───────────────────────────────────────────────────────

class TrainingSubject:
    """
    Manages observer subscriptions and broadcasts events.
    Trainer inherits/composes this. Models are fully decoupled from observers.
    """

    def __init__(self):
        self._observers: List[TrainingObserver] = []

    def attach(self, observer: TrainingObserver) -> None:
        """Subscribe an observer to all future events."""
        self._observers.append(observer)
        logger.debug("[TrainingSubject] Attached: %s", observer.observer_name)

    def detach(self, observer: TrainingObserver) -> None:
        """Unsubscribe an observer."""
        self._observers.remove(observer)

    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a named event + payload to all attached observers.
        Observer errors are caught and logged — one bad observer won't
        block the rest of the pipeline.
        """
        event = TrainingEvent(event_type=event_type, data=data)
        for obs in self._observers:
            try:
                obs.update(event)
            except Exception as exc:
                logger.error(
                    "[TrainingSubject] Observer '%s' raised: %s",
                    obs.observer_name, exc,
                )

    @property
    def observer_count(self) -> int:
        return len(self._observers)
