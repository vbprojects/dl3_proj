from __future__ import annotations

from typing import Final

STATUS_PENDING: Final[str] = "pending"
STATUS_RUNNING: Final[str] = "running"
STATUS_DONE: Final[str] = "done"
STATUS_FAILED: Final[str] = "failed"

ALL_STATUSES: Final[set[str]] = {
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_FAILED,
}

INDEX_COLUMNS: Final[list[str]] = [
    "image_sha",
    "label",
    "status",
    "attempts",
    "updated_at",
    "error",
    "conversation_json_path",
    "image_webp_path",
    "source_split",
    "source_idx",
]