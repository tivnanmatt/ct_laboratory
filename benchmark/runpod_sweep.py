#!/usr/bin/env python3
"""Fan out the uint16-only projector benchmark across RunPod GPU types.

For each GPU type: create a pod (SSH-enabled entrypoint), wait for SSH, stream
benchmark progress, scp the results JSON, and ALWAYS delete the pod when done
or on any failure. Runs GPU types in parallel threads.

Usage:
    python3 runpod_sweep.py                  # full default GPU list
    python3 runpod_sweep.py "NVIDIA GeForce RTX 4090" "NVIDIA L40S"   # subset
"""
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

API_BASE = "https://rest.runpod.io/v1"
SSH_KEY = "/home/staticct/.runpod/ssh/runpodctl-ssh-key"
RESULTS_DIR = Path("/tmp/claude-1000/-home-staticct/d28c98e3-d700-4f07-9578-06a43c736637/scratchpad/results")
IMAGE = "tivnanmatt/private:ct-lab-benchmark"
BENCH_ARGS = "--resolutions 64 128 256 512 --backends cuda --precompute-modes compressed --repeats 3"

# On-demand-available types (checked 2026-07-04); "Reserved" types excluded:
# RTX 5090, B200, H200, L4, L40, A100-SXM4-40GB, RTX 2000 Ada.
DEFAULT_GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",        # $0.34/hr
    "NVIDIA GeForce RTX 5080",        # $0.39/hr
    "NVIDIA A40",                     # $0.35/hr
    "NVIDIA RTX 6000 Ada Generation", # $0.74/hr
    "NVIDIA L40S",                    # $0.79/hr
    "NVIDIA A100 80GB PCIe",          # $1.19/hr
    "NVIDIA A100-SXM4-80GB",          # $1.39/hr
    "NVIDIA H100 PCIe",               # $1.99/hr
    "NVIDIA H100 NVL",                # $2.59/hr
    "NVIDIA H100 80GB HBM3",          # $2.69/hr (SXM)
]

SSH_UP_TIMEOUT_S = 20 * 60      # image pull + boot
BENCH_TIMEOUT_S = 40 * 60       # sweep after SSH is up
POLL_S = 20


def api_key():
    txt = open(Path.home() / ".runpod" / "config.toml").read()
    return re.search(r'apikey\s*=\s*["\']([^"\']+)["\']', txt, re.I).group(1)


KEY = api_key()


def api(method, path, body=None):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode() or "{}")


def registry_auth_id():
    auths = api("GET", "/containerregistryauth")
    return auths[0]["id"]


def slug(gpu_type):
    return re.sub(r"[^a-z0-9]+", "-", gpu_type.lower().replace("nvidia", "").strip()).strip("-")


def ssh_run(ip, port, cmd, timeout=25):
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", f"ConnectTimeout=10",
         "-i", SSH_KEY, "-p", str(port), f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def log(s, msg):
    print(f"[{s}] {msg}", flush=True)


def bench_one(gpu_type, reg_auth):
    s = slug(gpu_type)
    tag = f"runpod-{s}"
    pod_id = None
    try:
        pod = api("POST", "/pods", {
            "name": f"ctlab-bench-{s}",
            "imageName": IMAGE,
            "gpuTypeIds": [gpu_type],
            "gpuCount": 1,
            "cloudType": "SECURE",
            "containerDiskInGb": 40,
            "ports": ["22/tcp"],
            "containerRegistryAuthId": reg_auth,
            # image is torch+cu130: host driver must support CUDA >= 13.0
            "allowedCudaVersions": ["13.0"],
            "dockerEntrypoint": ["bash", "-lc",
                'ssh-keygen -A && mkdir -p ~/.ssh && echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys '
                '&& chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && /usr/sbin/sshd && '
                f'python /opt/benchmark/projector_timing_benchmark.py {BENCH_ARGS} --tag {tag} '
                '2>&1 | tee /opt/benchmark/bench.log; sleep infinity'],
        })
        pod_id = pod["id"]
        log(s, f"pod {pod_id} created (${pod.get('costPerHr','?')}/hr, {pod.get('machine',{}).get('dataCenterId','?')})")

        # Wait for SSH
        ip = port = None
        deadline = time.time() + SSH_UP_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(POLL_S)
            try:
                # runpodctl ssh info is the proven source of ip/port
                r = subprocess.run(
                    [str(Path.home() / ".local/bin/runpodctl"), "ssh", "info", pod_id],
                    capture_output=True, text=True, timeout=30)
                d = json.loads(r.stdout or "{}")
                pip = d.get("ip") or ""
                pport = d.get("port")
                if pip and pport:
                    rc, out = ssh_run(pip, pport, "echo UP")
                    if rc == 0 and "UP" in out:
                        ip, port = pip, pport
                        break
            except Exception:
                pass
        if not ip:
            raise RuntimeError("SSH never came up")
        log(s, f"ssh up at {ip}:{port}, streaming benchmark")

        # Wait for benchmark completion, reporting progress
        seen = 0
        deadline = time.time() + BENCH_TIMEOUT_S
        done = False
        while time.time() < deadline:
            time.sleep(POLL_S)
            try:
                rc, out = ssh_run(ip, port, "grep -E '\\[bench\\]|RESULTS_JSON_END' /opt/benchmark/bench.log 2>/dev/null | grep -v RESULTS_JSON_BEGIN")
            except Exception:
                continue
            lines = [l for l in out.splitlines() if l.strip()]
            for l in lines[seen:]:
                if "RESULTS_JSON_END" not in l:
                    log(s, l)
            seen = len([l for l in lines if "RESULTS_JSON_END" not in l])
            if any("RESULTS_JSON_END" in l for l in lines):
                done = True
                break
        if not done:
            raise RuntimeError("benchmark timed out")

        # Collect
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = RESULTS_DIR / f"{tag}.json"
        r = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=accept-new", "-i", SSH_KEY, "-P", str(port),
             f"root@{ip}:/opt/benchmark/output/sweep_{tag}/projector_timing.json", str(dest)],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"scp failed: {r.stderr[:200]}")
        log(s, f"results collected -> {dest}")
        return {"gpu": gpu_type, "status": "ok", "file": str(dest)}
    except Exception as e:
        log(s, f"FAILED: {e}")
        return {"gpu": gpu_type, "status": f"failed: {e}"}
    finally:
        if pod_id:
            try:
                api("DELETE", f"/pods/{pod_id}")
                log(s, f"pod {pod_id} deleted")
            except Exception as e:
                log(s, f"!! POD DELETE FAILED ({pod_id}): {e} — DELETE MANUALLY")


def main():
    gpu_types = sys.argv[1:] or DEFAULT_GPU_TYPES
    reg_auth = registry_auth_id()
    print(f"Sweeping {len(gpu_types)} GPU types (uint16-only, res 64-512), registry auth {reg_auth}", flush=True)
    results = []
    threads = []
    lock = threading.Lock()

    def worker(gt):
        r = bench_one(gt, reg_auth)
        with lock:
            results.append(r)

    for gt in gpu_types:
        t = threading.Thread(target=worker, args=(gt,))
        t.start()
        threads.append(t)
        time.sleep(3)  # stagger creations
    for t in threads:
        t.join()

    print("\n=== SWEEP SUMMARY ===", flush=True)
    for r in sorted(results, key=lambda x: x["gpu"]):
        print(f"{r['gpu']:<35} {r['status']}", flush=True)


if __name__ == "__main__":
    main()
