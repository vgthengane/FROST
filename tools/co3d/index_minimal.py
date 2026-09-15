#!/usr/bin/env python3
"""Helpers and orchestration for the CO3D FSCIL index-minimal download."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_index(repo: Path) -> Path:
    candidates = [
        repo / "third_party/microshape/data/index_files/co3d",
        repo.parent / "SelfEvo-CL3D/third_party/microshape/data/index_files/co3d",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _needed_sequences(cat: str, index_root: Path) -> set[str]:
    needed: set[str] = set()
    for split in ("train", "test"):
        for session_file in (index_root / split).glob("session_*.txt"):
            for line in session_file.read_text().splitlines():
                if not line.strip():
                    continue
                rel = line.split()[0].replace("v10-co3d/", "")
                parts = rel.split("/")
                if len(parts) == 3 and parts[0] == cat:
                    needed.add(parts[2].removesuffix(".pt"))
    return needed


def load_categories(links_json: Path, index_root: Path) -> list[str]:
    links = json.loads(links_json.read_text())
    index_cats = set()
    for split in ("train", "test"):
        for session_file in (index_root / split).glob("session_*.txt"):
            for line in session_file.read_text().splitlines():
                if not line.strip():
                    continue
                rel = line.split()[0].replace("v10-co3d/", "")
                parts = rel.split("/")
                if len(parts) >= 1 and parts[0]:
                    index_cats.add(parts[0])

    ordered: list[str] = []
    for cat in links.get("full", {}):
        if cat.upper() == "METADATA":
            continue
        if cat in index_cats:
            ordered.append(cat)
    for cat in sorted(index_cats - set(ordered)):
        ordered.append(cat)
    return ordered


def count_expected_zips(links_json: Path, cat: str) -> int:
    links = json.loads(links_json.read_text())["full"]
    return len(links.get(cat, []))


def category_complete(cat: str, cat_dir: Path, index_root: Path) -> tuple[bool, str]:
    needed = _needed_sequences(cat, index_root)
    if not needed:
        return False, f"{cat}: index has no entries, treating as incomplete"

    present = {
        seq
        for seq in needed
        if (cat_dir / seq / "pointcloud.ply").is_file()
    }
    missing = needed - present
    empty_dirs = 0
    if cat_dir.is_dir():
        for seq_dir in cat_dir.iterdir():
            if seq_dir.is_dir() and not (seq_dir / "pointcloud.ply").is_file():
                empty_dirs += 1

    msg = (
        f"{cat}: needed={len(needed)} pointclouds={len(present)} "
        f"missing={len(missing)} empty_dirs={empty_dirs}"
    )
    return len(missing) == 0, msg


def cleanup_empty_sequence_dirs(cat_dir: Path) -> int:
    """Remove sequence folders that have no pointcloud.ply (stale placeholders)."""
    removed = 0
    if not cat_dir.is_dir():
        return removed
    for seq_dir in list(cat_dir.iterdir()):
        if seq_dir.is_dir() and not (seq_dir / "pointcloud.ply").is_file():
            shutil.rmtree(seq_dir, ignore_errors=True)
            removed += 1
    return removed


def backup_existing_pointclouds(cat_dir: Path, backup_dir: Path) -> int:
    """Copy existing pointcloud.ply files aside before a full re-download."""
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up = 0
    if not cat_dir.is_dir():
        return backed_up
    for seq_dir in cat_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        src = seq_dir / "pointcloud.ply"
        if src.is_file():
            dst_dir = backup_dir / seq_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "pointcloud.ply")
            backed_up += 1
    return backed_up


def prune_category(
    cat: str,
    cat_dir: Path,
    tmp: Path,
    index_root: Path,
    *,
    backup_dir: Path | None = None,
) -> str:
    needed = _needed_sequences(cat, index_root)
    kept: list[str] = []
    restored: list[str] = []
    missing: list[str] = []

    for seq in sorted(needed):
        dst_dir = tmp / seq
        src = cat_dir / seq / "pointcloud.ply"
        if src.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "pointcloud.ply")
            kept.append(seq)
            continue

        if backup_dir is not None:
            backup_src = backup_dir / seq / "pointcloud.ply"
            if backup_src.is_file():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_src, dst_dir / "pointcloud.ply")
                restored.append(seq)
                continue

        missing.append(seq)

    lines = [
        f"{cat}: needed={len(needed)} kept={len(kept)} restored={len(restored)} missing={len(missing)}",
    ]
    if missing[:5]:
        lines.append(f"  missing examples: {missing[:5]}")
    return "\n".join(lines)


def collect_class_stats(
    cat: str,
    out: Path,
    index_root: Path,
    expected_zips: int,
) -> dict[str, int | str]:
    cat_dir = out / cat
    needed = _needed_sequences(cat, index_root)

    folders = 0
    pointclouds = 0
    empty = 0
    if cat_dir.is_dir():
        for seq_dir in cat_dir.iterdir():
            if seq_dir.is_dir():
                folders += 1
                if (seq_dir / "pointcloud.ply").is_file():
                    pointclouds += 1
                else:
                    empty += 1

    final_zips = len(list(out.glob(f"{cat}_*.zip")))
    inprog_dir = out / "_in_progress"
    inprog_zips = len(list(inprog_dir.glob(f"{cat}_*.zip"))) if inprog_dir.is_dir() else 0
    downloaded_zips = expected_zips - final_zips - inprog_zips

    return {
        "category": cat,
        "expected_zips": expected_zips,
        "downloaded_zips": downloaded_zips,
        "extracted_zips": expected_zips,
        "index_sequences": len(needed),
        "kept_folders": folders,
        "pointclouds": pointclouds,
        "empty_folders": empty,
    }


@dataclass
class Config:
    repo: Path
    out: Path
    index: Path
    links_json: Path
    download_script: Path
    log: Path
    run_log: Path
    class_stats_log: Path
    python: str
    max_retries: int
    retry_wait_seconds: int
    n_download_workers: int
    n_extract_workers: int
    start_from: str | None
    only_category: str | None


def _resolve_python() -> str:
    for key in ("PYTHON_BIN", "PYTHON"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    micromamba = Path("/user/HS502/vt00262/micromamba/envs/co3d-dataset/bin/python")
    if micromamba.is_file():
        return str(micromamba)
    for candidate in ("python3", "python"):
        found = shutil.which(candidate)
        if found:
            return found
    return sys.executable


def load_config() -> Config:
    repo = _repo_root()
    run_id = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = repo / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    retry_wait = int(
        os.environ.get("RETRY_WAIT_SECONDS")
        or os.environ.get("RETRY_SLEEP_SEC")
        or "20"
    )

    out = os.environ.get("OUT", "").strip() or str(repo / "_datasets/co3d")
    index = os.environ.get("INDEX", "").strip() or str(_default_index(repo))
    links_json = os.environ.get("LINKS_JSON", "").strip() or str(repo / "co3d/links.json")

    return Config(
        repo=repo,
        out=Path(out),
        index=Path(index),
        links_json=Path(links_json),
        download_script=repo / "co3d/download_dataset.py",
        log=logs / "co3d_index_minimal_download.log",
        run_log=logs / f"co3d_index_minimal_download_{run_id}.log",
        class_stats_log=logs / f"co3d_class_stats_{run_id}.csv",
        python=_resolve_python(),
        max_retries=int(os.environ.get("MAX_RETRIES", "100")),
        retry_wait_seconds=retry_wait,
        n_download_workers=int(os.environ.get("N_DOWNLOAD_WORKERS", "4")),
        n_extract_workers=int(os.environ.get("N_EXTRACT_WORKERS", "4")),
        start_from=os.environ.get("START_FROM") or None,
        only_category=os.environ.get("ONLY_CATEGORY") or None,
    )


class Logger:
    def __init__(self, log: Path, run_log: Path) -> None:
        self.log = log
        self.run_log = run_log
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str, *, timestamp: bool = True) -> None:
        if timestamp:
            line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        else:
            line = message
        print(line, flush=True)
        for path in (self.log, self.run_log):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _append_subprocess_output(proc: subprocess.Popen[str], logger: Logger) -> int:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        for path in (logger.log, logger.run_log):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    return proc.wait()


def download_category(cfg: Config, logger: Logger, cat: str) -> None:
    expected_zips = count_expected_zips(cfg.links_json, cat)
    cat_dir = cfg.out / cat
    work_tmp = cfg.out / "_keep_tmp" / cat
    backup_dir = cfg.out / "_keep_tmp" / f"{cat}_backup"

    removed_empty = cleanup_empty_sequence_dirs(cat_dir)
    if removed_empty:
        logger.write(f"{cat}: removed {removed_empty} empty sequence folders before download")

    backed_up = backup_existing_pointclouds(cat_dir, backup_dir)
    logger.write(f"{cat}: backed up {backed_up} existing pointcloud.ply files")

    attempt = 1
    while True:
        logger.write(f"Download start: {cat} (attempt {attempt})")
        proc = subprocess.Popen(
            [
                cfg.python,
                str(cfg.download_script),
                "--download_folder",
                str(cfg.out),
                "--download_categories",
                cat,
                "--n_download_workers",
                str(cfg.n_download_workers),
                "--n_extract_workers",
                str(cfg.n_extract_workers),
                "--clear_archives_after_unpacking",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = _append_subprocess_output(proc, logger)
        if rc == 0:
            break
        if attempt >= cfg.max_retries:
            raise RuntimeError(
                f"Download failed for {cat} after {attempt} attempts (exit {rc})."
            )
        logger.write(
            f"Download failed for {cat} (exit {rc}). "
            f"Retrying in {cfg.retry_wait_seconds}s..."
        )
        import time

        time.sleep(cfg.retry_wait_seconds)
        attempt += 1

    logger.write(f"Prune start: {cat}")
    if work_tmp.exists():
        shutil.rmtree(work_tmp, ignore_errors=True)
    work_tmp.mkdir(parents=True, exist_ok=True)
    for line in prune_category(cat, cat_dir, work_tmp, cfg.index, backup_dir=backup_dir).splitlines():
        logger.write(line, timestamp=False)

    shutil.rmtree(cat_dir, ignore_errors=True)
    cat_dir.mkdir(parents=True, exist_ok=True)
    if work_tmp.is_dir() and any(work_tmp.iterdir()):
        for item in work_tmp.iterdir():
            shutil.move(str(item), str(cat_dir / item.name))

    shutil.rmtree(cfg.out / "_keep_tmp", ignore_errors=True)
    for zip_path in cfg.out.glob(f"{cat}_*.zip"):
        zip_path.unlink(missing_ok=True)

    complete, msg = category_complete(cat, cat_dir, cfg.index)
    logger.write(msg, timestamp=False)
    if not complete:
        logger.write(f"Warning: {cat} still incomplete after download/prune")

    stats = collect_class_stats(cat, cfg.out, cfg.index, expected_zips)
    _append_stats_row(cfg.class_stats_log, stats)
    logger.write(f"Stats written for {cat} -> {cfg.class_stats_log}")
    logger.write(f"Done: {cat}")


def _append_stats_row(path: Path, row: dict[str, int | str]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_all(cfg: Config) -> None:
    logger = Logger(cfg.log, cfg.run_log)
    cfg.out.mkdir(parents=True, exist_ok=True)

    categories = load_categories(cfg.links_json, cfg.index)
    if cfg.only_category:
        if cfg.only_category not in categories:
            raise RuntimeError(f"ONLY_CATEGORY={cfg.only_category!r} not in index categories")
        categories = [cfg.only_category]
    if not categories:
        raise RuntimeError(f"No categories found for index at {cfg.index}")

    with cfg.class_stats_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "expected_zips",
                "downloaded_zips",
                "extracted_zips",
                "index_sequences",
                "kept_folders",
                "pointclouds",
                "empty_folders",
            ],
        )
        writer.writeheader()

    logger.write(f"CO3D index-minimal download start -> {cfg.out}")
    logger.write(f"Run log: {cfg.run_log}")
    logger.write(f"Class stats log: {cfg.class_stats_log}")
    logger.write(
        "Retry policy: "
        f"MAX_RETRIES={cfg.max_retries} RETRY_WAIT_SECONDS={cfg.retry_wait_seconds}"
    )
    if cfg.only_category:
        logger.write(f"Only category: {cfg.only_category}")
    if cfg.start_from:
        logger.write(f"Resuming from category: {cfg.start_from}")

    skip = bool(cfg.start_from) and not cfg.only_category
    for cat in categories:
        if skip:
            if cat == cfg.start_from:
                skip = False
            else:
                logger.write(f"Skip (before resume point): {cat}")
                continue

        removed_empty = cleanup_empty_sequence_dirs(cfg.out / cat)
        if removed_empty:
            logger.write(f"{cat}: removed {removed_empty} empty sequence folders")

        complete, msg = category_complete(cat, cfg.out / cat, cfg.index)
        logger.write(msg, timestamp=False)
        if complete:
            logger.write(f"Skip (already complete): {cat}")
            stats = collect_class_stats(
                cat, cfg.out, cfg.index, count_expected_zips(cfg.links_json, cat)
            )
            _append_stats_row(cfg.class_stats_log, stats)
            for zip_path in cfg.out.glob(f"{cat}_*.zip"):
                zip_path.unlink(missing_ok=True)
            continue

        download_category(cfg, logger, cat)

    logger.write("All categories finished")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run",),
        help="Run the index-minimal download pipeline.",
    )
    args = parser.parse_args()
    if args.command == "run":
        cfg = load_config()
        run_all(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
