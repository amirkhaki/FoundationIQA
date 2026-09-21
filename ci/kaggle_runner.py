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


def find_iqa_data_root(base=pathlib.Path("/kaggle/input"), max_depth=5):
    """Find the pre-extracted pyiqa dataset tree among the attached inputs, wherever Kaggle
    mounted it (the mount layout differs between dataset / notebook-output sources, so
    guessing the path is fragile). Markers: PREPARED_WITH.txt (written by
    ci/kaggle_prepare_datasets.py), or pyiqa's own meta_info/ folder. Returns a list,
    marker-file matches first. Bounded depth: never descends into the image folders."""
    found = []

    def visit(d, depth):
        if (d / "PREPARED_WITH.txt").is_file() or (d / "meta_info").is_dir():
            found.append(d)
            return  # don't look inside a data root
        if depth >= max_depth:
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in children:
            visit(c, depth + 1)

    if base.exists():
        visit(base, 0)
    return sorted(found, key=lambda d: not (d / "PREPARED_WITH.txt").is_file())


def scratch_dir():
    """Big, writable, NOT part of the kernel output."""
    for c in ("/kaggle/temp", "/tmp"):
        if os.path.isdir(c) and os.access(c, os.W_OK):
            return pathlib.Path(c)
    return pathlib.Path("/tmp")


def count_files(path):
    return sum(len(names) for _, _, names in os.walk(path))


def extract_iqa_tars(src, dest, command):
    """The prepared inputs hold one .tar per dataset (Kaggle caps notebook output at ~500 files, so an
    extracted tree cannot be stored). Extract what the command needs into `dest` (local disk) and verify
    the file counts against MANIFEST.json. A dataset counts as needed if its name appears in the command
    (case-insensitive); if the command names none of them (e.g. the default dataset list), extract all.
    meta_info is always extracted. Returns the list of extracted names."""
    tars = sorted(src.glob("*.tar"))
    manifest_file = src / "MANIFEST.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    cmd = command.lower()
    wanted = [t for t in tars if t.stem != "meta_info" and t.stem.lower() in cmd] or \
             [t for t in tars if t.stem != "meta_info"]
    wanted += [t for t in tars if t.stem == "meta_info"]
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for t in wanted:
        if (dest / t.stem).exists():
            print(f"already extracted: {t.stem}", flush=True)
            continue
        t1 = time.time()
        sh(["tar", "-xf", str(t), "-C", str(dest)])
        got = count_files(dest / t.stem)
        want = manifest.get(t.stem, {}).get("files")
        print(f"extracted {t.stem}: {got} files in {time.time() - t1:.0f}s (manifest: {want})", flush=True)
        if want is not None and got != want:
            raise RuntimeError(f"{t.stem}: extracted {got} files but the manifest says {want} - "
                               "the prepared datasets are damaged, re-run the prep notebook")
        done.append(t.stem)
    return done


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
    # under /kaggle/input. Where exactly depends on the source type, so we search for the
    # pyiqa dataset tree instead of assuming a path, and hand it to the command as
    # IQA_DATA_ROOT (see step 4).
    inputs = pathlib.Path("/kaggle/input")
    meta["inputs"] = sorted(p.name for p in inputs.iterdir()) if inputs.exists() else []
    print("attached inputs:", meta["inputs"], flush=True)
    roots = find_iqa_data_root()
    data_root = roots[0] if roots else None
    if data_root:
        print(f"IQA data source: {data_root}", flush=True)
        if len(roots) > 1:
            print(f"WARNING: several candidates {[str(r) for r in roots]}; using the first", flush=True)
        if any(data_root.glob("*.tar")):  # tar layout (see ci/kaggle_prepare_datasets.py)
            t1 = time.time()
            meta["iqa_extracted"] = extract_iqa_tars(data_root, scratch_dir() / "iqa_datasets", cfg["command"])
            meta["iqa_extract_seconds"] = round(time.time() - t1, 1)
            data_root = scratch_dir() / "iqa_datasets"
        print(f"IQA data root: {data_root}", flush=True)
    else:
        print("WARNING: no prepared IQA datasets found under /kaggle/input "
              "(looked for PREPARED_WITH.txt / meta_info/). pyiqa will DOWNLOAD the datasets. "
              "Input layout (dirs, depth<=4):", flush=True)
        subprocess.run("find /kaggle/input -maxdepth 4 -type d 2>/dev/null | head -60", shell=True)
    meta["iqa_data_root"] = str(data_root) if data_root else None

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
    # Where pyiqa should look for the datasets. Precedence: an explicit --data-root in the
    # command > IQA_DATA_ROOT already set in the environment > the tree found in step 2 >
    # scratch/iqa_datasets (a download target that can't ship back as output).
    if "IQA_DATA_ROOT" not in env:
        env["IQA_DATA_ROOT"] = str(data_root) if data_root else str(scratch_dir() / "iqa_datasets")
    print("IQA_DATA_ROOT for the command:", env["IQA_DATA_ROOT"], flush=True)
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
