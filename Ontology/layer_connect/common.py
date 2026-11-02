from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml


RELEVANCE_ORDER = {"context": 0, "supporting": 1, "core": 2}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def unique_text(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_text(value)
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def clamp(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "\x1f".join(normalize_text(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_json_records(path: Path, keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        for key in keys:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path}: JSON object 목록이 필요합니다.")
    return payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: 잘못된 JSONL") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: JSON object가 아닙니다.")
            yield item


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: config root는 object여야 합니다.")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def config_path(config: dict[str, Any], key: str, section: str = "paths") -> Path:
    raw = config.get(section, {}).get(key)
    if not raw:
        raise KeyError(f"config.{section}.{key}가 필요합니다.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def output_path(config: dict[str, Any], filename: str) -> Path:
    return config_path(config, "output_dir") / filename


def strongest_relevance(first: str, second: str) -> str:
    first = first if first in RELEVANCE_ORDER else "context"
    second = second if second in RELEVANCE_ORDER else "context"
    return max((first, second), key=RELEVANCE_ORDER.__getitem__)


def lca_hid(hids: Iterable[str], parent_by_hid: dict[str, str | None]) -> str | None:
    clean = [hid for hid in hids if hid in parent_by_hid]
    if not clean:
        return None

    def lineage(hid: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        current: str | None = hid
        while current is not None and current not in seen:
            seen.add(current)
            values.append(current)
            current = parent_by_hid.get(current)
        return list(reversed(values))

    paths = [lineage(hid) for hid in clean]
    common: str | None = None
    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        common = parts[0]
    return common


def is_ancestor(
    ancestor: str, descendant: str, parent_by_hid: dict[str, str | None]
) -> bool:
    current: str | None = descendant
    seen: set[str] = set()
    while current is not None and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        current = parent_by_hid.get(current)
    return False

