# Production deploy checklist (text-to-icpc2.com)

The model is on the HF Hub (`diogocarapito/text-to-icpc2`, tag `v1`). The server downloads it on first start, so nothing large has to be copied to the server.

## 1. Server

- Ubuntu 24.04, **4 GB RAM minimum** (8 GB comfortable), 2+ vCPU, 25 GB disk. The app container is capped at 3 GB. The model is 1.1 GB (fp32) and runs on the CPU.
- Firewall: allow 22/tcp, 80/tcp, 443/tcp and 443/udp (HTTP/3).
- Docker Engine + compose plugin: `curl -fsSL https://get.docker.com | sh`

## 2. DNS

At the domain registrar, point both names at the server IP. Caddy requests the TLS certificate only after they resolve.

| type | name | value |
|---|---|---|
| A | `text-to-icpc2.com` | server IPv4 |
| A | `www` | server IPv4 (redirected to the apex) |
| AAAA | both | server IPv6, if it has one |

Check with `dig +short text-to-icpc2.com`.

## 3. Code and secrets

```bash
git clone https://github.com/diogocarapito/text-to-icpc2 && cd text-to-icpc2/deploy
nano .env && chmod 600 .env
```

`deploy/.env` (never commit it):

```
DOMAIN=text-to-icpc2.com
ACME_EMAIL=<your email, used by Let's Encrypt only>
POSTGRES_PASSWORD=<openssl rand -hex 24>
ICPC2_ADMIN_TOKEN=<openssl rand -hex 32>   # same value in your Mac's .env, for `icpc2 pull-feedback`
HF_TOKEN=<write token>                      # used by the scheduler (private feedback repo, promotions); optional until 5
MODEL_REVISION=v1                           # pin; switch to "main" once automatic promotion is live
# DIGITALOCEAN_TOKEN=                       # leave out until the GPU retrain path is fixed (see 5)
```

## 4. Start

```bash
docker compose up -d --build db app caddy backup     # everything except the scheduler (see 5)
docker compose logs -f app                           # first start downloads the model (~1 min)
curl https://text-to-icpc2.com/health                # {"model":"v1",...}
```

- Dashboard (hidden, not linked): `https://text-to-icpc2.com/painel`. Enter `ICPC2_ADMIN_TOKEN` once; a cookie keeps that browser signed in for 30 days. Changing the token signs every browser out. The token is accepted only in the form or an `Authorization: Bearer` header, never in the URL.
- **Correções por rever** (top of the dashboard): user corrections to a code other than the 1st suggestion. Nothing enters the live index or training until you approve it. Feedback you give on the main page while signed in counts as approval.
- Export for retraining (from the Mac): `ICPC2_VPS_URL=https://text-to-icpc2.com icpc2 pull-feedback`

## 5. Scheduler (automatic retrains): hold for now

The `scheduler` container runs `icpc2 loop --backend do` daily. That path is **not production-ready**: the DigitalOcean `gpu-h100x1-base` droplet booted without an NVIDIA driver on 2026-11-22 (see JOURNAL). Until that is fixed:
- feedback is still logged in Postgres, and backed up nightly by `backup`;
- retrains run from the Mac: `icpc2 pull-feedback && icpc2 loop` (local backend), which promotes to the HF Hub and calls `/api/admin/reload` on the server.

When the GPU path works: add `DIGITALOCEAN_TOKEN`, set `MODEL_REVISION=main`, then `docker compose up -d scheduler`. Its SSH key is generated in a volume on first start and registered with DigitalOcean automatically.

## 6. Operations

| task | command |
|---|---|
| update code | `./deploy.sh` from the repo root (backs up the DB, `git pull`, rebuilds the app, waits for `/health`) |
| roll back the model | set `MODEL_REVISION=vN` in `.env`, then `docker compose up -d app` |
| backups | `deploy/backups/icpc2-YYYY-MM-DD.sql.gz`, 14 days. Copy them off the server (e.g. a weekly `rsync` to the Mac). |
| restore | `gunzip -c backups/<file>.sql.gz \| docker compose exec -T db psql -U icpc2user icpc2` |
| logs | `docker compose logs --tail 100 app` |

**Privacy:** Caddy keeps no access log. The app stores the typed text (e-mails, full dates and long/grouped digit runs masked as `<EMAIL>`/`<DATA>`/`<NUM>`; names are not detected), predictions, feedback and a random per-browser session id (30-day cookie). It stores no IP addresses and no accounts. User text leaves the server only to the **private** HF feedback repo.

## Not yet verified

- `docker compose build` has not been run (Docker was not available on the Mac). The local app was tested with `icpc2 serve` against the same HF model. If the build fails, send the log.
- Cold-start RAM and CPU latency on the actual server size.
