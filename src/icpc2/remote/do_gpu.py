"""Ephemeral DigitalOcean GPU Droplets for training.

    icpc2 gpu run -- icpc2 sweep configs/sweeps/benchmark-01.yaml
    icpc2 gpu list | icpc2 gpu cleanup

Lifecycle (GPU Droplets bill while powered off, so they are always DESTROYED):
  create droplet (AI/ML image, your DO SSH keys) -> wait for SSH -> clone this repo at the current
  pushed commit -> pip install -> run the command -> rsync experiments/ (+ best model) back -> destroy.
A self-destruct timer on the droplet deletes it after `max_hours` even if this machine sleeps.

Needs DIGITALOCEAN_TOKEN (droplet create/read/delete, ssh_key read) and an SSH key in your DO account
whose private key is loaded on this machine.
"""

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

import httpx

from icpc2.config import load_config, path

log = logging.getLogger(__name__)
API = "https://api.digitalocean.com/v2"
TAG = "icpc2-train"


def _client() -> httpx.Client:
    token = os.environ.get("DIGITALOCEAN_TOKEN")
    if not token:
        raise SystemExit("DIGITALOCEAN_TOKEN not set (add it to .env)")
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {token}"}, timeout=60)


def list_droplets() -> list[dict]:
    with _client() as c:
        r = c.get("/droplets", params={"tag_name": TAG, "per_page": 100})
        r.raise_for_status()
        return r.json()["droplets"]


def destroy(droplet_id: int):
    with _client() as c:
        r = c.delete(f"/droplets/{droplet_id}")
        if r.status_code not in (204, 404):
            r.raise_for_status()
    log.info("droplet %s destroyed", droplet_id)


def cleanup() -> int:
    ds = list_droplets()
    for d in ds:
        destroy(d["id"])
    return len(ds)


def sizes() -> list[dict]:
    with _client() as c:
        r = c.get("/sizes", params={"per_page": 200})
        r.raise_for_status()
        return [s for s in r.json()["sizes"] if s["slug"].startswith("gpu-")]


def _self_destruct(max_hours: float) -> str:
    """cloud-init user_data: delete this droplet via the API after max_hours, no matter what."""
    token = os.environ["DIGITALOCEAN_TOKEN"]
    secs = int(max_hours * 3600)
    return f"""#!/bin/bash
ID=$(curl -s http://169.254.169.254/metadata/v1/id)
systemd-run --on-active={secs} --unit=icpc2-self-destruct \\
  curl -s -X DELETE -H "Authorization: Bearer {token}" {API}/droplets/$ID
"""


def _git_ref() -> str:
    def git(*a):
        return subprocess.check_output(["git", *a], cwd=path("."), text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Uncommitted changes: commit + push first (the droplet clones the public repo)")
    sha = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", sha):  # no remote branch contains HEAD
        raise SystemExit("HEAD is not pushed to origin: push first")
    return sha


def _ssh_base(ip: str) -> list[str]:
    kh = path("data/.gpu_known_hosts")
    key = os.environ.get("ICPC2_SSH_KEY")  # set on the server's scheduler; locally your agent's keys
    return ["ssh", *(["-i", key] if key else []), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={kh}", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", f"root@{ip}"]


def _ensure_key(c: httpx.Client) -> None:
    """Register ICPC2_SSH_KEY's public half with DigitalOcean (scheduler container)."""
    key = os.environ.get("ICPC2_SSH_KEY")
    if not key:
        return
    pub = Path(key + ".pub").read_text().strip()
    existing = [k["public_key"].strip() for k in c.get("/account/keys").json()["ssh_keys"]]
    if pub not in existing:
        r = c.post("/account/keys", json={"name": "icpc2-scheduler", "public_key": pub})
        r.raise_for_status()
        log.info("registered scheduler SSH key with DigitalOcean")


def _wait_ssh(ip: str, timeout: int = 900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if subprocess.run(_ssh_base(ip) + ["true"], capture_output=True).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError("SSH did not come up")


def create(candidates: list[str], region: str | None, image: str, max_hours: float) -> tuple[int, str]:
    """Create the first (size, region) DigitalOcean actually accepts. /v2/sizes lists where a size
    is offered, not live capacity, so a 422 'not available' just moves on to the next option."""
    with _client() as c:
        _ensure_key(c)
        keys = [k["id"] for k in c.get("/account/keys").json()["ssh_keys"]]
        if not keys:
            raise SystemExit("No SSH keys in your DigitalOcean account (Settings -> Security)")
        offered = {s["slug"]: s["regions"] for s in sizes()}
        did = None
        for size in candidates:
            for reg in ([region] if region else offered.get(size, [])):
                body = {"name": f"icpc2-train-{int(time.time())}", "region": reg, "size": size,
                        "image": image, "ssh_keys": keys, "tags": [TAG], "monitoring": True,
                        "user_data": _self_destruct(max_hours)}
                r = c.post("/droplets", json=body)
                if r.status_code == 422:
                    log.info("%s in %s: %s", size, reg, r.json().get("message"))
                    continue
                if r.status_code >= 400:
                    raise SystemExit(f"create failed: {r.status_code} {r.text}")
                did = r.json()["droplet"]["id"]
                log.info("droplet %s (%s, %s) created; waiting for IP", did, size, reg)
                break
            if did:
                break
        if not did:
            raise SystemExit(f"No capacity for any of {candidates} right now; try again later")
        for _ in range(120):
            d = c.get(f"/droplets/{did}").json()["droplet"]
            ips = [n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public"]
            if d["status"] == "active" and ips:
                return did, ips[0]
            time.sleep(10)
    destroy(did)
    raise TimeoutError("droplet never became active")


BOOTSTRAP = """set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v rsync >/dev/null || apt-get install -y -qq rsync
# the AI/ML image ships Python 3.10; uv fetches a standalone Python 3.12 (and installs fast)
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
rm -rf /root/app && git clone -q {repo} /root/app && cd /root/app && git checkout -q {sha}
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e ".[train]"
.venv/bin/python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), 'torch', torch.__version__)"
"""


def run(command: list[str], size: str | None = None, region: str | None = None,
        fetch: tuple[str, ...] = ("experiments", "reports"), fetch_best_model: bool = True,
        keep: bool = False) -> int:
    cfg = load_config("loop")["gpu"]
    candidates = [size] if size else cfg["sizes"]
    sha = _git_ref()
    did, ip = create(candidates, region, cfg["image"], cfg["max_hours"])
    log_path = path("reports") / f"gpu-{did}.log"
    log_path.parent.mkdir(exist_ok=True)
    rc = 1
    try:
        _wait_ssh(ip)
        ssh = _ssh_base(ip)
        boot = BOOTSTRAP.format(repo=cfg["repo"], sha=sha)
        subprocess.run(ssh + ["bash", "-lc", shlex.quote(boot)], check=True)
        env = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in
                       ("HF_TOKEN", "ICPC2_ADMIN_TOKEN", "ICPC2_VPS_URL", "WANDB_API_KEY")
                       if os.environ.get(k))
        remote = f"cd /root/app && {env} .venv/bin/icpc2 build-data && {env} " + \
                 " ".join(shlex.quote(c) for c in command).replace("icpc2 ", ".venv/bin/icpc2 ", 1)
        log.info("running on %s: %s", ip, " ".join(command))
        with open(log_path, "w") as lf:
            p = subprocess.Popen(ssh + ["bash", "-lc", shlex.quote(remote)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lf.write(line)
                if not any(s in line for s in ("it/s]", "s/it]")):
                    print(line, end="")
            rc = p.wait()
        for d in fetch:
            subprocess.run(["rsync", "-az", "-e", " ".join(ssh[:-1]), f"root@{ip}:/root/app/{d}/",
                            str(path(d)) + "/"], check=False)
        if fetch_best_model:
            for best in path("experiments").glob("*/best_model.txt"):
                for remote_dir in best.read_text().split():
                    local = Path(remote_dir.replace("/root/app/", str(path(".")) + "/"))
                    local.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["rsync", "-az", "-e", " ".join(ssh[:-1]),
                                    f"root@{ip}:{remote_dir}/", str(local) + "/"], check=False)
                best.write_text("\n".join(p.replace("/root/app/", "") for p in best.read_text().split()))
    finally:
        if keep:
            log.warning("--keep: droplet %s (%s) left running; it self-destructs after %sh",
                        did, ip, cfg["max_hours"])
        else:
            destroy(did)
    log.info("exit code %s; log at %s", rc, log_path)
    return rc
