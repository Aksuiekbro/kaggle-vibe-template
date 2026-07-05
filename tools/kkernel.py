#!/usr/bin/env python3
"""Remote training runner: execute workspace scripts on Kaggle's free compute.

The droplet is the brain; Kaggle kernels are the muscle (~4 vCPU / ~30 GB RAM
CPU sessions, GPU sessions under the weekly quota). Maps onto C13: probes run
locally at low fidelity, promoted full-fidelity runs ship here.

Usage:
  python tools/kkernel.py run    --script path.py --title my-exp [--competition slug]
                                 [--gpu] [--out dir] [--timeout-min 120]
  python tools/kkernel.py push   --script path.py --title my-exp [--competition slug] [--gpu]
  python tools/kkernel.py status --ref user/slug
  python tools/kkernel.py output --ref user/slug --out dir

Notes:
- The script must write everything it wants to keep (submission.csv, oof.csv,
  metrics.json) to the kernel working dir; `output` downloads it plus the log.
- Competition data mounts at /kaggle/input/<competition>/ inside the kernel.
- GPU sessions burn the ~30h/week quota — request --gpu only when the model
  needs it (C13 applies to hardware too).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

POLL_SECONDS = 30


def sh(args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def kaggle_username():
    result = sh(["kaggle", "config", "view"])
    match = re.search(r"username:\s*(\S+)", result.stdout)
    if not match:
        sys.exit("ERROR: could not determine Kaggle username (kaggle config view). "
                 "Is the CLI authenticated?")
    return match.group(1)


def slugify(title):
    slug = re.sub(r"[^a-z0-9-]+", "-", title.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "experiment"


def build_kernel_dir(script, title, username, competition=None, datasets=None, gpu=False):
    workdir = Path(tempfile.mkdtemp(prefix="kkernel-"))
    code_name = "script.py"
    shutil.copy2(script, workdir / code_name)
    slug = slugify(title)
    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_internet": "false",
        "competition_sources": [competition] if competition else [],
        "dataset_sources": datasets or [],
        "kernel_sources": [],
    }
    with open(workdir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return workdir, meta["id"]


def cmd_push(args):
    ref = push(args)
    print(f"PUSHED: {ref}")
    print(f"  poll:   python tools/kkernel.py status --ref {ref}")
    print(f"  fetch:  python tools/kkernel.py output --ref {ref} --out <dir>")
    return 0


def push(args):
    script = Path(args.script)
    if not script.exists():
        sys.exit(f"ERROR: script not found: {script}")
    username = kaggle_username()
    workdir, ref = build_kernel_dir(
        script, args.title, username,
        competition=args.competition, datasets=args.dataset, gpu=args.gpu,
    )
    result = sh(["kaggle", "kernels", "push", "-p", str(workdir)])
    shutil.rmtree(workdir, ignore_errors=True)
    out = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "error" in out.lower():
        sys.exit(f"ERROR: push failed:\n{out}")
    print(out)
    return ref


def kernel_status(ref):
    result = sh(["kaggle", "kernels", "status", ref])
    out = (result.stdout + result.stderr).strip()
    match = re.search(r'status\s+"?([A-Za-z_]+)"?', out)
    return (match.group(1).lower() if match else "unknown"), out


def cmd_status(args):
    status, raw = kernel_status(args.ref)
    print(raw)
    return 0 if status not in ("error", "cancelacknowledged") else 1


def cmd_output(args):
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    result = sh(["kaggle", "kernels", "output", args.ref, "-p", str(dest)], timeout=900)
    print((result.stdout + result.stderr).strip())
    files = [p.name for p in dest.iterdir()] if dest.exists() else []
    print(f"downloaded: {files}")
    return result.returncode


def cmd_run(args):
    ref = push(args)
    print(f"PUSHED: {ref} — polling every {POLL_SECONDS}s (timeout {args.timeout_min} min)")
    deadline = time.time() + args.timeout_min * 60
    last = ""
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        status, _ = kernel_status(ref)
        if status != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {status}")
            last = status
        if status in ("complete", "kernelworkercompleted"):
            break
        if status in ("error", "cancelacknowledged", "kernelworkererror"):
            print("REMOTE RUN FAILED — fetching log for diagnosis...")
            break
    else:
        print("TIMEOUT waiting for kernel — it may still be running; "
              f"check later: python tools/kkernel.py status --ref {ref}")

    out_dir = Path(args.out or f"kernel-output-{slugify(args.title)}")
    args_out = argparse.Namespace(ref=ref, out=str(out_dir))
    cmd_output(args_out)
    log = out_dir / f"{ref.split('/')[1]}.log"
    if log.exists():
        print("--- last log lines ---")
        print("\n".join(log.read_text().splitlines()[-15:]))
    return 0 if last in ("complete", "kernelworkercompleted") else 1


def main():
    parser = argparse.ArgumentParser(description="Run scripts on Kaggle's free compute")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--script", required=True)
        p.add_argument("--title", required=True)
        p.add_argument("--competition", help="mount this competition's data at /kaggle/input/")
        p.add_argument("--dataset", action="append", help="dataset ref(s) to mount")
        p.add_argument("--gpu", action="store_true", help="request GPU (burns weekly quota)")

    p = sub.add_parser("push"); common(p); p.set_defaults(func=cmd_push)

    p = sub.add_parser("run"); common(p)
    p.add_argument("--out", help="output dir (default kernel-output-<slug>)")
    p.add_argument("--timeout-min", type=int, default=120)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status")
    p.add_argument("--ref", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("output")
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_output)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
