from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from lvm_utils.cache_schema import (
    INDEX_COLUMNS,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from lvm_utils.utils import hash_pil_image, materialize_conversation_images, serialize_conversation_with_image_refs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FirstStageCache:
    """Disk-backed cache for first-stage conversations keyed by image SHA256."""

    def __init__(
        self,
        cache_root: str | Path = "./cache_data",
        webp_quality: int = 80,
        running_ttl_seconds: int = 1800,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.webp_quality = webp_quality
        self.running_ttl_seconds = running_ttl_seconds

        self.index_path = self.cache_root / "index.parquet"
        self.images_dir = self.cache_root / "images"
        self.conversations_dir = self.cache_root / "conversations"

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._records = {}
            self._write_index_atomic()
            return

        table = pq.read_table(self.index_path)
        records = table.to_pylist()
        loaded: dict[str, dict[str, Any]] = {}
        for record in records:
            image_sha = record.get("image_sha")
            if not image_sha:
                continue
            normalized = {col: record.get(col) for col in INDEX_COLUMNS}
            normalized["attempts"] = int(normalized.get("attempts") or 0)
            loaded[image_sha] = normalized
        self._records = loaded

    def _write_index_atomic(self) -> None:
        rows = [self._records[k] for k in sorted(self._records.keys())]

        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.table({col: [] for col in INDEX_COLUMNS})

        with tempfile.NamedTemporaryFile(
            prefix="index-",
            suffix=".parquet",
            dir=self.cache_root,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        pq.write_table(table, tmp_path)
        tmp_path.replace(self.index_path)

    def _upsert(self, image_sha: str, patch: dict[str, Any]) -> dict[str, Any]:
        base = self._records.get(
            image_sha,
            {
                "image_sha": image_sha,
                "label": None,
                "status": STATUS_PENDING,
                "attempts": 0,
                "updated_at": _utc_now_iso(),
                "error": None,
                "conversation_json_path": None,
                "image_webp_path": None,
                "source_split": None,
                "source_idx": None,
            },
        )
        base.update(patch)
        base["updated_at"] = _utc_now_iso()
        self._records[image_sha] = base
        self._write_index_atomic()
        return base

    def stats(self) -> dict[str, int]:
        counts = {
            STATUS_PENDING: 0,
            STATUS_RUNNING: 0,
            STATUS_DONE: 0,
            STATUS_FAILED: 0,
        }
        for rec in self._records.values():
            status = rec.get("status")
            if status in counts:
                counts[status] += 1
        return counts

    def recover_stale_running(self) -> int:
        now = datetime.now(timezone.utc)
        moved = 0
        for image_sha, rec in list(self._records.items()):
            if rec.get("status") != STATUS_RUNNING:
                continue
            updated_at = rec.get("updated_at")
            if not updated_at:
                continue
            try:
                dt = datetime.fromisoformat(updated_at)
            except ValueError:
                continue
            if now - dt > timedelta(seconds=self.running_ttl_seconds):
                moved += 1
                self._records[image_sha] = {
                    **rec,
                    "status": STATUS_PENDING,
                    "updated_at": _utc_now_iso(),
                }
        if moved:
            self._write_index_atomic()
        return moved

    def mark_running(
        self,
        image_sha: str,
        source_split: str | None,
        source_idx: int | None,
        label: int | str | None,
    ) -> None:
        rec = self._records.get(image_sha)
        attempts = 1 if rec is None else int(rec.get("attempts") or 0) + 1
        self._upsert(
            image_sha,
            {
                "status": STATUS_RUNNING,
                "attempts": attempts,
                "error": None,
                "label": label,
                "source_split": source_split,
                "source_idx": source_idx,
            },
        )

    def mark_failed(self, image_sha: str, error: str) -> None:
        self._upsert(
            image_sha,
            {
                "status": STATUS_FAILED,
                "error": error,
            },
        )

    def load_done_conversation(self, image_sha: str) -> list[dict[str, Any]] | None:
        rec = self._records.get(image_sha)
        if not rec or rec.get("status") != STATUS_DONE:
            return None
        conv_path = rec.get("conversation_json_path")
        if not conv_path:
            return None
        abs_conv = self.cache_root / conv_path
        if not abs_conv.exists():
            return None
        payload = json.loads(abs_conv.read_text(encoding="utf-8"))
        return materialize_conversation_images(payload, self.cache_root)

    def save_done_conversation(
        self,
        image_sha: str,
        conversation: list[dict[str, Any]],
        source_split: str | None,
        source_idx: int | None,
        label: int | str | None,
    ) -> None:
        serial, first_image_rel = serialize_conversation_with_image_refs(
            conversation,
            self.cache_root,
            webp_quality=self.webp_quality,
        )
        conv_rel = Path("conversations") / f"{image_sha}.json"
        conv_abs = self.cache_root / conv_rel
        conv_abs.write_text(json.dumps(serial, ensure_ascii=False), encoding="utf-8")

        self._upsert(
            image_sha,
            {
                "status": STATUS_DONE,
                "error": None,
                "label": label,
                "conversation_json_path": conv_rel.as_posix(),
                "image_webp_path": first_image_rel,
                "source_split": source_split,
                "source_idx": source_idx,
            },
        )

    def get_or_compute(
        self,
        image: Image.Image,
        compute_fn,
        source_split: str | None = None,
        source_idx: int | None = None,
        label: int | str | None = None,
    ) -> tuple[list[dict[str, Any]], bool, str]:
        """Return conversation, hit flag, and image SHA."""
        image_sha = hash_pil_image(image)
        cached = self.load_done_conversation(image_sha)
        if cached is not None:
            # Backfill label metadata for already-cached rows when available.
            if label is not None:
                self._upsert(image_sha, {"label": label})
            return cached, True, image_sha

        self.mark_running(
            image_sha,
            source_split=source_split,
            source_idx=source_idx,
            label=label,
        )
        try:
            conversation = compute_fn(image)
            self.save_done_conversation(
                image_sha,
                conversation,
                source_split=source_split,
                source_idx=source_idx,
                label=label,
            )
            return conversation, False, image_sha
        except Exception as exc:
            self.mark_failed(image_sha, error=str(exc))
            raise
    def get_or_compute_batch(
        self,
        images: list[Image.Image],
        compute_batch_fn,
        source_splits: list[str | None] | None = None,
        source_idxs: list[int | None] | None = None,
        labels: list[int | str | None] | None = None,
    ) -> tuple[list[list[dict[str, Any]]], list[bool], list[str]]:
        """Batch evaluate cache hits and missing computations."""
        shas = [hash_pil_image(img) for img in images]
        
        results = [None] * len(images)
        hits = [False] * len(images)
        
        misses = []
        
        for i, sha in enumerate(shas):
            lbl = labels[i] if labels else None
            cached = self.load_done_conversation(sha)
            if cached is not None:
                if lbl is not None:
                    self._upsert(sha, {"label": lbl})
                results[i] = cached
                hits[i] = True
            else:
                misses.append(i)
                self.mark_running(
                    sha,
                    source_split=source_splits[i] if source_splits else None,
                    source_idx=source_idxs[i] if source_idxs else None,
                    label=lbl,
                )
                
        if misses:
            try:
                miss_images = [images[i] for i in misses]
                miss_convs = compute_batch_fn(miss_images)
                for j, miss_idx in enumerate(misses):
                    sha = shas[miss_idx]
                    conv = miss_convs[j]
                    self.save_done_conversation(
                        sha,
                        conv,
                        source_split=source_splits[miss_idx] if source_splits else None,
                        source_idx=source_idxs[miss_idx] if source_idxs else None,
                        label=labels[miss_idx] if labels else None,
                    )
                    results[miss_idx] = conv
            except Exception as exc:
                for j, miss_idx in enumerate(misses):
                    self.mark_failed(shas[miss_idx], error=str(exc))
                raise exc

        return results, hits, shas
