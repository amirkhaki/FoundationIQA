# ci/kaggle_runner.py
#
# This file is the body of the Kaggle *script kernel*. The GitHub workflow
# prepends a `RUN_CONFIG = {...}` dict (repo, sha, command, ...) and pushes the
# result to Kaggle, so everything below can assume RUN_CONFIG exists.
#
# The command to run is NOT defined here: it comes from the workflow input
# (RUN_CONFIG["command"]). This file only does the plumbing around it: clone the
# exact commit, install, run the command, record run_meta.json.
#
# Outputs: everything left in /kaggle/working is downloaded by the workflow and
# uploaded as a GitHub artifact, so point big files (e.g. --out-dir for the
# Tier-1 caches) at /tmp if you don't want them back.
import datetime
import importlib.metadata as md
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

WORK = pathlib.Path("/kaggle/working")
SRC = pathlib.Path("/tmp/FoundationIQA")
META = WORK / "run_meta.json"

cfg = RUN_CONFIG  # noqa: F821
meta = {
    "github_run_id": cfg["github_run_id"],  # the workflow uses this to recognise *its* run
    "repo": cfg["repo"],
    "commit": cfg["sha"],
    "ref": cfg["ref"],
    "command": cfg["command"],
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "started",
}


def write_meta():
    META.write_text(json.dumps(meta, indent=2))


def sh(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


write_meta()  # written first, so even an early failure leaves a matching meta behind
t0 = time.time()
try:
    # --- 1. get the exact commit -------------------------------------------
    # Public repo: plain clone. For a PRIVATE repo this fails; see the notes in
    # the workflow (embed `git archive` output instead of cloning).
    sh(["git", "clone", f"https://github.com/{cfg['repo']}.git", str(SRC)])
    sh(["git", "checkout", "--detach", cfg["sha"]], cwd=SRC)
    meta["commit_verified"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SRC, text=True
    ).strip()

    # --- 2. inputs ---------------------------------------------------------------
    # Datasets / kernels attached via dataset_sources / kernel_sources appear read-only
    # under /kaggle/input/<slug>. Listing them makes the exact --data-root path visible.
    inputs = pathlib.Path("/kaggle/input")
    meta["inputs"] = sorted(p.name for p in inputs.iterdir()) if inputs.exists() else []
    print("attached inputs:", meta["inputs"], flush=True)

    # --- 3. install + record the environment ---------------------------------
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(SRC)])
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
    except FileNotFoundError:  # CPU session
        gpu = ""
    meta["gpu"] = gpu or None
    meta["torch"] = version("torch")
    meta["pyiqa"] = version("pyiqa")

    # --- 4. run the command from the workflow --------------------------------
    # cwd=/kaggle/working, so files written to cwd (master_results.csv) become outputs.
    # The interpreter we pip-installed into goes first on PATH, so `python` / `pip`
    # in the command mean the same environment. -e / pipefail: stop at the first
    # failing command and don't let `cmd | tee` hide a failure.
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    # If the command doesn't say where the IQA datasets live (--data-root beats this),
    # keep any download OUT of /kaggle/working so it can't ship back as output.
    env.setdefault("IQA_DATA_ROOT", "/tmp/iqa_datasets")
    print("$", cfg["command"], flush=True)
    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", cfg["command"]],
        cwd=WORK, env=env, check=True,
    )
    meta["status"] = "ok"
except BaseException as e:  # noqa: BLE001 - record anything, then re-raise so Kaggle marks the run as failed
    meta["status"] = "failed"
    meta["error"] = repr(e)
    raise
finally:
    # Safety net for anything that still used pyiqa's default ./datasets (= /kaggle/working/datasets
    # here): it would come back as GBs of "output" and end up in a GitHub artifact.
    stray = WORK / "datasets"
    if stray.exists():
        print(f"WARNING: removing {stray} so it is not shipped as output. "
              "Use --data-root / IQA_DATA_ROOT to keep datasets elsewhere.", flush=True)
        shutil.rmtree(stray, ignore_errors=True)
        meta["removed_stray_datasets_dir"] = True
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["wall_seconds"] = round(time.time() - t0, 1)
    write_meta()
