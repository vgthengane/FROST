#!/usr/bin/env python3
"""Compare FSCIL index files against on-disk CO3D sequence folders."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_index_line(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if not line:
        return None
    rel = line.split()[0].replace("v10-co3d/", "")
    parts = rel.split("/")
    if len(parts) != 3:
        return None
    cat, split, seq = parts
    return cat, split, seq.removesuffix(".pt")


def load_session_file(path: Path) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        parsed = parse_index_line(line)
        if parsed:
            entries.append(parsed)
    return entries


def disk_inventory(dataset_root: Path) -> set[tuple[str, str]]:
    on_disk: set[tuple[str, str]] = set()
    for cat_dir in dataset_root.iterdir():
        if not cat_dir.is_dir():
            continue
        for seq_dir in cat_dir.iterdir():
            if seq_dir.is_dir() and (seq_dir / "pointcloud.ply").is_file():
                on_disk.add((cat_dir.name, seq_dir.name))
    return on_disk


def main() -> None:
    repo = Path(__file__).resolve().parent
    index_root = repo / "third_party/microshape/data/index_files/co3d"
    dataset_root = repo / "_datasets/co3d"
    out_dir = dataset_root / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    on_disk = disk_inventory(dataset_root)
    index_sets: dict[str, set[tuple[str, str]]] = {"train": set(), "test": set()}
    per_session: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    all_rows: list[dict[str, str]] = []

    for split in ("train", "test"):
        for session_file in sorted((index_root / split).glob("session_*.txt")):
            session = session_file.stem.replace("session_", "")
            entries = load_session_file(session_file)
            per_session[(split, session)] = entries
            for cat, _sp, seq in entries:
                index_sets[split].add((cat, seq))
                status = "present" if (cat, seq) in on_disk else "missing"
                all_rows.append(
                    {
                        "split": split,
                        "session": session,
                        "class": cat,
                        "sequence": seq,
                        "status": status,
                        "expected_path": f"{cat}/{seq}/pointcloud.ply",
                    }
                )

    all_index = index_sets["train"] | index_sets["test"]
    class_rows: list[dict[str, int | str]] = []
    for cat in sorted({c for c, _ in all_index}):
        needed = {(c, s) for c, s in all_index if c == cat}
        train_needed = {(c, s) for c, s in index_sets["train"] if c == cat}
        test_needed = {(c, s) for c, s in index_sets["test"] if c == cat}
        class_rows.append(
            {
                "class": cat,
                "train_needed": len(train_needed),
                "train_present": len(train_needed & on_disk),
                "train_missing": len(train_needed - on_disk),
                "test_needed": len(test_needed),
                "test_present": len(test_needed & on_disk),
                "test_missing": len(test_needed - on_disk),
                "total_needed": len(needed),
                "total_present": len(needed & on_disk),
                "total_missing": len(needed - on_disk),
            }
        )

    session_stats = []
    for (split, session), entries in sorted(per_session.items()):
        needed = {(c, s) for c, _sp, s in entries}
        present = needed & on_disk
        session_stats.append(
            {
                "split": split,
                "session": int(session),
                "needed": len(needed),
                "present": len(present),
                "missing": len(needed - present),
                "coverage_pct": round(100 * len(present) / len(needed), 2) if needed else 100.0,
            }
        )

    split_summary = {}
    for split in ("train", "test"):
        needed = index_sets[split]
        present = needed & on_disk
        missing = needed - on_disk
        missing_by_class = defaultdict(int)
        for cat, _seq in missing:
            missing_by_class[cat] += 1
        split_summary[split] = {
            "needed": len(needed),
            "present": len(present),
            "missing": len(missing),
            "coverage_pct": round(100 * len(present) / len(needed), 2) if needed else 100.0,
            "missing_by_class": dict(sorted(missing_by_class.items())),
        }

    with (out_dir / "index_coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": split_summary["train"], "test": split_summary["test"], "sessions": session_stats}, handle, indent=2)
        handle.write("\n")

    with (out_dir / "index_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    with (out_dir / "index_all_entries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    missing_rows = [row for row in all_rows if row["status"] == "missing"]
    with (out_dir / "index_missing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(missing_rows)

    print(f"Dataset root: {dataset_root}")
    for split in ("train", "test"):
        s = split_summary[split]
        print(
            f"{split.upper():5}  present {s['present']}/{s['needed']}  "
            f"missing {s['missing']}  ({s['coverage_pct']}%)"
        )
    print(f"\nMissing entries written: {len(missing_rows)} rows -> {out_dir / 'index_missing.csv'}")
    worst = sorted(class_rows, key=lambda row: -int(row["total_missing"]))[:10]
    print("\nMost missing classes:")
    for row in worst:
        if int(row["total_missing"]) == 0:
            break
        print(
            f"  {row['class']}: {row['total_missing']} missing "
            f"(train {row['train_missing']}, test {row['test_missing']})"
        )


if __name__ == "__main__":
    main()
