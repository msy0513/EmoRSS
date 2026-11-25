import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class Pair:
    core_id: str
    source: str
    risk: str
    neutral_prompt: str
    high_prompt: str


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number} contains invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number} must contain a JSON object")
                yield row


def index_by_core(path):
    rows = {}
    for row in read_jsonl(path):
        key = row.get("core_id")
        if not isinstance(key, str) or not key.strip() or key in rows:
            raise ValueError(f"{path} contains a missing or duplicate core_id")
        rows[key] = row
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_pairs(path):
    rows = index_by_core(path)
    pairs = []
    for key in sorted(rows):
        row = rows[key]
        fields = ("source", "risk", "neutral_prompt", "high_prompt")
        if any(not isinstance(row.get(name), str) or not row[name].strip() for name in fields):
            raise ValueError(f"{path} has missing fields for core {key}")
        if row["risk"] not in {"low", "high"}:
            raise ValueError(f"{path} has an invalid risk label for core {key}")
        pairs.append(Pair(key, *(row[name] for name in fields)))
    return pairs


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


@contextmanager
def output_text(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_jsonl(path, rows):
    with output_text(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
