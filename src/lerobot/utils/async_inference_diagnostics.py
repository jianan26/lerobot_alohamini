"""Non-blocking diagnostic recording for asynchronous inference."""

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Any

import cv2
import numpy as np
import torch

DEFAULT_DIAGNOSTICS_DIR = Path("logs/async_diagnostics")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_diagnostics_dir(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repository_root() / path


def should_sample(request_index: int, interval: int) -> bool:
    """Sample the first request and every ``interval`` requests after it."""
    return request_index >= 1 and (request_index - 1) % interval == 0


def _snapshot(value: Any) -> Any:
    """Detach mutable/device-backed values before handing them to the writer thread."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value))
    if isinstance(value, dict):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class DiagnosticRecorder:
    """Write JSONL events and sampled observations without blocking control threads."""

    def __init__(
        self,
        component: str,
        *,
        enabled: bool = False,
        root_dir: str | Path = DEFAULT_DIAGNOSTICS_DIR,
        session_id: str | None = None,
        manifest: dict[str, Any] | None = None,
        queue_size: int = 256,
    ) -> None:
        self.enabled = enabled
        self.component = component
        self.dropped_items = 0
        self._closed = False
        self._queue: Queue[tuple[str, Any] | None] | None = None
        self._thread: threading.Thread | None = None

        if not enabled:
            self.output_dir = None
            return

        session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        root = resolve_diagnostics_dir(root_dir)
        self.output_dir = root / session_id / component
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "samples").mkdir(exist_ok=True)

            manifest_payload = {
                "component": component,
                "session_id": session_id,
                "output_dir": str(self.output_dir),
                "pid": os.getpid(),
                "started_at": time.time(),
                **(manifest or {}),
            }
            (self.output_dir / "manifest.json").write_text(
                json.dumps(_snapshot(manifest_payload), indent=2, ensure_ascii=False) + "\n"
            )
            (root / f"latest_{component}.json").write_text(
                json.dumps(
                    {"session_id": session_id, "output_dir": str(self.output_dir)},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        except OSError:
            logging.getLogger(__name__).exception(
                "Could not initialize %s diagnostics at %s; diagnostics disabled",
                component,
                self.output_dir,
            )
            self.enabled = False
            self.output_dir = None
            return

        self._queue = Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logging.getLogger(__name__).info("%s diagnostics: %s", component, self.output_dir)

    def record_event(self, event: str, **payload: Any) -> None:
        if not self.enabled or self._closed:
            return
        item = {
            "component": self.component,
            "event": event,
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            **payload,
        }
        self._enqueue(("event", _snapshot(item)))

    def record_observation_sample(
        self,
        request_id: int,
        *,
        state: dict[str, Any],
        images: dict[str, np.ndarray],
    ) -> None:
        if not self.enabled or self._closed:
            return
        image_copies = {name: np.asarray(image).copy() for name, image in images.items()}
        self._enqueue(("sample", (request_id, _snapshot(state), image_copies)))

    def _enqueue(self, item: tuple[str, Any]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(item)
        except Full:
            self.dropped_items += 1
            if self.dropped_items == 1:
                logging.getLogger(__name__).warning(
                    "%s diagnostic queue is full; dropping records", self.component
                )

    def _writer_loop(self) -> None:
        assert self._queue is not None
        assert self.output_dir is not None
        events_path = self.output_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as events_file:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    kind, payload = item
                    if kind == "event":
                        events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        events_file.flush()
                    else:
                        self._write_sample(*payload)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to write %s diagnostic record", self.component
                    )
                finally:
                    self._queue.task_done()

    def _write_sample(self, request_id: int, state: dict[str, Any], images: dict[str, np.ndarray]) -> None:
        assert self.output_dir is not None
        sample_dir = self.output_dir / "samples" / f"request_{request_id:08d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, image in images.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
            if not cv2.imwrite(str(sample_dir / f"{safe_name}.jpg"), image):
                raise OSError(f"Failed to write diagnostic image {name}")

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        assert self._queue is not None
        assert self._thread is not None
        self.record_event("diagnostics_stopping", dropped_items=self.dropped_items)
        self._closed = True
        self._queue.put(None)
        self._thread.join()
