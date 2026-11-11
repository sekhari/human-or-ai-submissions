"""Train and evaluate every model in an experiment profile, in parallel across
GPUs, then build a report (results table, graphs, sanity checks, ensembles).

  python run_all.py --profile blends         # the members of the uploaded blends (~4 h, 2 GPUs)
  python run_all.py --profile blends --only lm_gpt2,lm_gpt2_medium
  python run_all.py --profile blends --skip ft_deberta_v3_large --set cv.n_folds=3
  python run_all.py --profile blends --gpus 0       # use only the first GPU
  python run_all.py --resume outputs/sessions/<session>   # finish an interrupted session
  python run_all.py --report outputs/sessions/<session>   # rebuild the report only
  python run_all.py --profile blends --dry-run       # show the plan

How it works
  * Profiles live in configs/experiments/<profile>.yaml (list of jobs: config, resource, cost,
    optional `set` overrides, `group`, `after`).
  * Each job is an ordinary `python train.py --model <config> ...` subprocess, so anything
    run_all does you can reproduce by hand. Its console output goes to
    outputs/sessions/<session>/logs/<job>.log.
  * One worker per GPU (CUDA_VISIBLE_DEVICES pins the job to that card), one per Apple MPS
    device, plus --cpu-workers CPU workers. Jobs are started heaviest-first; idle GPU workers
    also pick up CPU jobs.
  * Pretrained weights are downloaded once up front (scripts/download_models.py) so parallel
    jobs never download the same model twice.
  * The report goes to experiments/sessions/<session>/ (tracked in git: report.md,
    summary.csv, figures/) and the best submission to outputs/submissions/.
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import read_yaml, set_data_dir  # noqa: E402

JOB_FIELDS = ["name", "config", "resource", "group", "cost", "after", "slot", "status", "returncode",
              "started", "finished", "duration_s", "cv_auc", "run_name", "log", "error"]


# ------------------------------------------------------------------ hardware
def detect_slots(args) -> list[dict]:
    slots = []
    try:
        import torch
        if torch.cuda.is_available():
            ids = [int(i) for i in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
            for i in ids:
                p = torch.cuda.get_device_properties(i)
                slots.append({"id": f"cuda{i}", "kind": "gpu", "device": "cuda",
                              "env": {"CUDA_VISIBLE_DEVICES": str(i)},
                              "desc": f"GPU {i}: {p.name} ({p.total_memory / 1e9:.0f} GB)",
                              "speed": p.multi_processor_count})
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            slots.append({"id": "mps", "kind": "gpu", "device": "mps", "env": {}, "desc": "Apple GPU (MPS)",
                          "speed": 1})
    except ImportError:
        pass
    if args.fake_gpus:  # testing the scheduler on a machine without GPUs
        slots = [{"id": f"fake{i}", "kind": "gpu", "device": "cpu", "env": {"CUDA_VISIBLE_DEVICES": ""},
                  "desc": f"fake GPU {i} (runs on CPU)", "speed": 1} for i in range(args.fake_gpus)]
    slots.sort(key=lambda s: -s["speed"])  # fastest card picks first
    for i in range(args.cpu_workers):
        slots.append({"id": f"cpu{i}", "kind": "cpu", "device": "cpu", "env": {}, "desc": f"CPU worker {i}",
                      "speed": 0})
    return slots


# ------------------------------------------------------------------ jobs
def load_jobs(profile: dict, args) -> list[dict]:
    defaults = profile.get("defaults") or {}
    jobs = []
    for j in profile["jobs"]:
        sets = dict(defaults)
        sets.update(j.get("set") or {})
        jobs.append({"name": j["name"], "config": j.get("config", j["name"]),
                     "resource": j.get("resource", "cpu"), "cost": float(j.get("cost", 1)),
                     "group": j.get("group", "main"), "after": j.get("after", ""), "set": sets})
    names = {j["name"] for j in jobs}
    if args.only:
        keep = set(args.only.split(","))
        unknown = keep - names
        if unknown:
            sys.exit(f"--only: unknown job(s) {sorted(unknown)}. Jobs: {sorted(names)}")
        jobs = [j for j in jobs if j["name"] in keep]
    if args.skip:
        jobs = [j for j in jobs if j["name"] not in set(args.skip.split(","))]
    kept = {j["name"] for j in jobs}
    for j in jobs:
        if j["after"] and j["after"] not in kept:
            j["after"] = ""  # dependency not part of this session
        for kv in args.set:
            k, v = kv.split("=", 1)
            j["set"][k] = yaml.safe_load(v)
        if not (ROOT / "configs" / "models" / f"{j['config']}.yaml").exists():
            sys.exit(f"Job {j['name']}: configs/models/{j['config']}.yaml does not exist")
    return jobs


def set_args(sets: dict) -> list[str]:
    out = []
    for k, v in sets.items():
        if isinstance(v, (list, dict)):
            v = yaml.safe_dump(v, default_flow_style=True).strip()
        elif isinstance(v, bool):
            v = "true" if v else "false"
        out.append(f"{k}={v}")
    return out


def write_jobs_csv(session: Path, jobs: list[dict]) -> None:
    tmp = session / "jobs.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS, extrasaction="ignore")
        w.writeheader()
        for j in jobs:
            w.writerow({**j, "after": j.get("after", "")})
    os.replace(tmp, session / "jobs.csv")


def tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
        lines = [l for l in lines if l.strip() and "it/s]" not in l and "s/it]" not in l]
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return ""


# ------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, jobs, slots, session, args):
        self.jobs, self.slots, self.session, self.args = jobs, slots, session, args
        self.lock = threading.Lock()
        self.procs = {}
        self.stop = threading.Event()
        self.events = queue.Queue()

    def _deps_ok(self, j):
        if not j["after"]:
            return True
        dep = next((d for d in self.jobs if d["name"] == j["after"]), None)
        return dep is None or dep["status"] in ("ok", "failed", "skipped")

    def _next_job(self, slot):
        with self.lock:
            pending = [j for j in self.jobs if j["status"] == "pending" and self._deps_ok(j)]
            mine = [j for j in pending if j["resource"] == slot["kind"]]
            if not mine and slot["kind"] == "gpu":
                mine = [j for j in pending if j["resource"] == "cpu"]  # idle GPU worker helps with CPU jobs
            if not mine and not self.slots_of("gpu") and slot["kind"] == "cpu":
                mine = [j for j in pending if j["resource"] == "gpu"]  # no GPU at all: run on CPU
            if not mine:
                return None
            j = max(mine, key=lambda x: x["cost"])
            j["status"], j["slot"] = "running", slot["id"]
            return j

    def slots_of(self, kind):
        return [s for s in self.slots if s["kind"] == kind]

    def _remaining(self):
        with self.lock:
            return any(j["status"] in ("pending", "running") for j in self.jobs)

    def _run_job(self, slot, j):
        log = self.session / "logs" / f"{j['name']}.log"
        device = slot["device"] if j["resource"] == "gpu" else "cpu"
        cmd = [sys.executable, "train.py", "--model", j["config"], "--run-name", j["run_name"],
               "--device", device, "--notes", f"run_all:{self.args.profile_name}:{j['name']}"]
        if not self.args.log_results:
            cmd.append("--no-log")
        sets = set_args(j["set"])
        if sets:
            cmd += ["--set", *sets]
        env = os.environ.copy()
        env.update(slot["env"] if j["resource"] == "gpu" else {})
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TQDM_MININTERVAL": "15",
                    "TOKENIZERS_PARALLELISM": "false",
                    # Windows without Developer Mode: HF prints a symlink warning on every load,
                    # which buries the real error at the end of a failed job's log.
                    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
        j["started"] = datetime.now().strftime("%H:%M:%S")
        j["log"] = str(log.relative_to(ROOT))
        t0 = time.time()
        self.events.put(("start", j, slot))
        with open(log, "a", encoding="utf-8") as fh:  # append: a resumed job keeps earlier attempts' logs
            fh.write(f"\n===== attempt started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            fh.write("$ " + " ".join(cmd) + "\n\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            with self.lock:
                self.procs[j["name"]] = p
            rc = p.wait()
        with self.lock:
            self.procs.pop(j["name"], None)
            j["returncode"] = rc
            j["duration_s"] = round(time.time() - t0)
            j["finished"] = datetime.now().strftime("%H:%M:%S")
            st = read_yaml(ROOT / "outputs" / "runs" / j["run_name"] / "status.json") \
                if (ROOT / "outputs" / "runs" / j["run_name"] / "status.json").exists() else {}
            if rc == 0 and st.get("status") == "ok":
                j["status"], j["cv_auc"] = "ok", st.get("cv_auc")
            else:
                j["status"] = "interrupted" if self.stop.is_set() else "failed"
                j["error"] = tail(log, 12)
            write_jobs_csv(self.session, self.jobs)
        self.events.put(("end", j, slot))

    def _worker(self, slot):
        while not self.stop.is_set():
            j = self._next_job(slot)
            if j is None:
                if not self._remaining():
                    return
                time.sleep(1)
                continue
            self._run_job(slot, j)

    def run(self):
        from tqdm.auto import tqdm
        todo = [j for j in self.jobs if j["status"] == "pending"]
        threads = [threading.Thread(target=self._worker, args=(s,), daemon=True) for s in self.slots]
        for t in threads:
            t.start()
        bar = tqdm(total=len(todo), desc="jobs", unit="job", dynamic_ncols=True)
        try:
            while any(t.is_alive() for t in threads):
                try:
                    kind, j, slot = self.events.get(timeout=1)
                except queue.Empty:
                    continue
                if kind == "start":
                    tqdm.write(f"[{j['started']}] start  {j['name']:<30} on {slot['desc']}")
                else:
                    bar.update(1)
                    msg = (f"CV AUROC {j['cv_auc']:.4f}" if j["status"] == "ok" and j.get("cv_auc") is not None
                           else f"{j['status'].upper()} (see {j['log']})")
                    tqdm.write(f"[{j['finished']}] done   {j['name']:<30} {msg}  [{j['duration_s']}s]")
                    if j["status"] == "failed":
                        tqdm.write("    " + "\n    ".join(j["error"].splitlines()[-4:]))
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted - stopping running jobs (resume later with --resume)...")
            self.stop.set()
            with self.lock:
                for p in self.procs.values():
                    p.terminate()
            for t in threads:
                t.join(timeout=30)
            for j in self.jobs:
                if j["status"] in ("pending", "running"):
                    j["status"] = "interrupted" if j["status"] == "running" else "pending"
            write_jobs_csv(self.session, self.jobs)
        bar.close()


# ------------------------------------------------------------------ main
def prefetch(jobs, args):
    """Download every Hugging Face model the jobs will use (after their overrides), once,
    before the parallel phase -- so two jobs never download the same weights concurrently."""
    from src.config import load_config
    ids = set()
    for j in jobs:
        params = (load_config(j["config"], overrides=set_args(j["set"]))["model"].get("params") or {})
        for k in ("model_name", "lm_name", "performer_name"):
            v = params.get(k)
            if v and not Path(str(v)).exists():
                ids.add(str(v))
    if not ids:
        return
    print(f"Pre-downloading {len(ids)} pretrained model(s) (skip with --no-prefetch) ...")
    rc = subprocess.call([sys.executable, "scripts/download_models.py", "--ids", *sorted(ids)], cwd=ROOT)
    if rc != 0:
        print("WARNING: some downloads failed; the affected jobs will fail and be reported.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="configs/experiments/<profile>.yaml (blends | fixed | your own)")
    ap.add_argument("--only", help="comma-separated job names to run")
    ap.add_argument("--skip", help="comma-separated job names to skip")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override for every job")
    ap.add_argument("--gpus", help="comma-separated CUDA device indices to use (default: all)")
    ap.add_argument("--cpu-workers", type=int, default=1, help="parallel CPU-only jobs (default 1)")
    ap.add_argument("--resume", help="session directory to resume")
    ap.add_argument("--report", help="only (re)build the report for a session directory")
    ap.add_argument("--no-prefetch", action="store_true", help="don't pre-download Hugging Face weights")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake-gpus", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--data-dir", default=None, help="folder with the Kaggle CSVs (default: $HOA_DATA_DIR, ./data, common/data.py)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)
    # utf-8 + replace: job error tails can contain tqdm bar characters a cp1252 console can't encode
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    from src.report import build_report

    if args.report:
        build_report(Path(args.report).resolve())
        return

    if args.resume:
        session = Path(args.resume).resolve()
        meta = read_yaml(session / "session.yaml")
        args.profile_name = meta["profile"]
        with open(session / "jobs.csv", newline="", encoding="utf-8") as f:
            saved = {r["name"]: r for r in csv.DictReader(f)}
        jobs = meta["jobs"]
        for j in jobs:
            r = saved.get(j["name"], {})
            j.update({k: r.get(k, "") for k in JOB_FIELDS if k not in ("name", "config", "resource", "group", "cost", "after")})
            j["cost"] = float(j["cost"])
            if j.get("status") != "ok":
                j.update(status="pending", error="", returncode="", cv_auc="")
                for kv in args.set:  # e.g. a smaller batch size for a job that ran out of memory
                    k, v = kv.split("=", 1)
                    j["set"][k] = yaml.safe_load(v)
        if args.only or args.skip:  # restrict which unfinished jobs this resume runs
            keep = set(args.only.split(",")) if args.only else {j["name"] for j in jobs}
            keep -= set(args.skip.split(",")) if args.skip else set()
            for j in jobs:
                if j["status"] == "pending" and j["name"] not in keep:
                    j["status"] = "skipped"
    else:
        if not args.profile:
            ap.error("--profile (or --resume / --report) is required")
        profile = read_yaml(ROOT / "configs" / "experiments" / f"{args.profile}.yaml")
        args.profile_name = args.profile
        jobs = load_jobs(profile, args)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = ROOT / "outputs" / "sessions" / f"{stamp}_{args.profile}"
        for j in jobs:
            j.update(status="pending", run_name=f"{stamp}_{args.profile}__{j['name']}", slot="", returncode="",
                     started="", finished="", duration_s="", cv_auc="", log="", error="")
        meta = {"profile": args.profile, "created": stamp, "description": profile.get("description", ""),
                "log_results": profile.get("log_results", True),
                "report": profile.get("report", {}), "jobs": [{k: j[k] for k in
                ("name", "config", "resource", "group", "cost", "after", "set", "run_name")} for j in jobs]}

    args.log_results = meta.get("log_results", True)
    slots = detect_slots(args)
    print(f"Session: {session.relative_to(ROOT)}")
    print("Workers: " + "; ".join(s["desc"] for s in slots))
    if not any(s["kind"] == "gpu" for s in slots) and any(j["resource"] == "gpu" for j in jobs):
        print("WARNING: no GPU found - GPU jobs will run on the CPU worker (slow).")
    pending = [j for j in jobs if j["status"] == "pending"]
    print(f"Jobs: {len(pending)} to run, {len(jobs) - len(pending)} already done")
    for j in sorted(pending, key=lambda x: (x["resource"], -x["cost"])):
        extra = f" after {j['after']}" if j["after"] else ""
        print(f"  [{j['resource']}] {j['name']:<30} config={j['config']:<26} cost={j['cost']:g}{extra}")
    if args.dry_run:
        return

    (session / "logs").mkdir(parents=True, exist_ok=True)
    with open(session / "session.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    write_jobs_csv(session, jobs)

    if not args.no_prefetch and pending:
        prefetch(pending, args)

    t0 = time.time()
    Scheduler(jobs, slots, session, args).run()
    n_ok = sum(j["status"] == "ok" for j in jobs)
    n_fail = sum(j["status"] == "failed" for j in jobs)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min: {n_ok} ok, {n_fail} failed, "
          f"{len(jobs) - n_ok - n_fail} not run")
    build_report(session)


if __name__ == "__main__":
    main()
