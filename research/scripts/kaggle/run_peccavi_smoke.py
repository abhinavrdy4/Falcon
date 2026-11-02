#!/usr/bin/env python
"""Kaggle 2x T4 runner for configs/peccavi_smoke.yaml (called by peccavi_smoke.ipynb).

  python scripts/kaggle/run_peccavi_smoke.py --out /kaggle/working/peccavi_smoke [--gpus 2] [--config ...]
      [--no-gate] [--set dataset.max_images=10 ...]

- shard i of n runs on GPU i (CUDA_VISIBLE_DEVICES=i), output <out>/gpu<i>/<exp name>/
- GPU 0 first runs scripts/peccavi_check.py on its first image; if that check fails, GPU 1 is stopped and
  nothing else runs (exit 1). With --no-gate the check is still run and reported but does not stop the run.
- stages embed -> edit -> evaluate -> raw_l0 (detection on the pre-enhancement image, diagnostic) run as
  separate processes so each gets a wall-clock time; every stage skips outputs that already exist, so
  re-running (or resuming from a previous version's output) continues
- writes <out>/run_times.json, <out>/logs/*.log, then <out>/report/{rows.csv,results.md,figure}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def resume_from_inputs(out: Path, inputs: Path = Path("/kaggle/input")):
    """Copy a previous version's outputs (attached as notebook input) so finished work is skipped."""
    if not inputs.exists():
        return
    done: list[Path] = []
    for prev in sorted(inputs.glob(f"**/{out.name}"), key=lambda p: len(p.parts)):
        if any(d in prev.parents for d in done):  # gpu<i>/peccavi_smoke inside an output already copied
            continue
        if prev.is_dir() and prev.resolve() != out.resolve():
            done.append(prev)
            log(f"resuming from {prev}")
            shutil.copytree(prev, out, dirs_exist_ok=True, copy_function=_copy_if_missing)


def _copy_if_missing(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


class Shard(threading.Thread):
    def __init__(self, i: int, n: int, a, times: dict, abort: threading.Event, check_done: threading.Event):
        super().__init__(name=f"gpu{i}")
        self.i, self.n, self.a, self.times, self.abort, self.check_done = i, n, a, times, abort, check_done
        self.ok, self.proc = False, None

    def cmd(self, script: str, *extra: str) -> list[str]:
        return [sys.executable, script, "--config", self.a.config, "--set", *self.a.set,
                f"output_dir={self.a.out / f'gpu{self.i}'}", f"dataset.shard=[{self.i},{self.n}]", *extra]

    def stage(self, name: str, cmd: list[str]) -> bool:
        if self.abort.is_set():
            return False
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.i), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // self.n)))
        logf = self.a.out / "logs" / f"gpu{self.i}_{name}.log"
        log(f"[gpu{self.i}] {name} -> {logf}")
        t = time.time()
        with open(logf, "a") as f:
            self.proc = subprocess.Popen(cmd, cwd=RESEARCH, env=env, stdout=f, stderr=subprocess.STDOUT)
            rc = self.proc.wait()
        self.times.setdefault(f"gpu{self.i}", {})[f"{name}_s"] = round(time.time() - t, 1)
        log(f"[gpu{self.i}] {name} exit {rc} after {time.time() - t:.0f}s")
        return rc == 0

    def run(self):
        if self.i == 0:
            ok = self.stage("check", self.cmd("scripts/peccavi_check.py"))
            self.times["check_passed"] = ok
            if not ok and self.a.gate:
                self.abort.set()
            self.check_done.set()
            if not ok and self.a.gate:
                return
        for st in ("embed", "edit", "evaluate"):
            if not self.stage(st, self.cmd("run_experiment.py", "--stages", st)):
                return
        if not self.stage("raw_l0", self.cmd("scripts/peccavi_raw_l0.py")):
            return
        self.ok = True

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/peccavi_smoke"))
    ap.add_argument("--config", default="configs/peccavi_smoke.yaml")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--no-gate", dest="gate", action="store_false", help="report the 1-image check, don't stop on it")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides, e.g. dataset.max_images=10")
    a = ap.parse_args()
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    resume_from_inputs(a.out)
    times: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": sys.argv[1:]}
    abort, check_done = threading.Event(), threading.Event()
    shards = [Shard(i, a.gpus, a, times, abort, check_done) for i in range(a.gpus)]
    t0 = time.time()
    for s in shards:
        s.start()
    while any(s.is_alive() for s in shards):
        if abort.is_set():
            for s in shards:
                s.stop()
        time.sleep(5)
    times["total_wall_s"] = round(time.time() - t0, 1)
    (a.out / "run_times.json").write_text(json.dumps(times, indent=2))
    check = a.out / "gpu0" / "peccavi_smoke" / "peccavi_check.json"
    if abort.is_set():
        log("1-image check FAILED: watermarked L0 not detected or clean L0 detected. Stopped. See "
            f"{check} and {a.out / 'logs' / 'gpu0_check.log'}")
        sys.exit(1)
    exps = [a.out / f"gpu{i}" / "peccavi_smoke" for i in range(a.gpus)]
    rc = subprocess.call([sys.executable, "scripts/peccavi_report.py", "--exp", *map(str, exps),
                          "--out", str(a.out / "report"), "--runner-log", str(a.out / "run_times.json")], cwd=RESEARCH)
    failed = [s.name for s in shards if not s.ok]
    if failed or rc:
        log(f"finished with errors: shards {failed}, report exit {rc}; see {a.out / 'logs'}")
        sys.exit(1)
    log(f"done in {times['total_wall_s'] / 60:.0f} min -> {a.out / 'report' / 'results.md'}")


if __name__ == "__main__":
    main()
