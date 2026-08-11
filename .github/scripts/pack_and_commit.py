import base64
import glob
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GH_TOKEN = os.environ["GH_TOKEN"]
REPO = os.environ["REPO"]
BASE = f"https://api.github.com/repos/{REPO}"
BRANCH = "main"
ROOT = "artifact"


def api(method, url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {GH_TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    if body:
        req.add_header("Content-Type", "application/json")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504, 403, 429, 409):
                time.sleep(20 * (attempt + 1))
                continue
            return e.code, e.read()
        except Exception as e:
            time.sleep(20 * (attempt + 1))
    return -1, b""


def put_gz(path_in_repo, raw_bytes):
    gz = gzip.compress(raw_bytes, 9)
    code, cur = api("GET", f"{BASE}/contents/{path_in_repo}")
    payload = {
        "message": f"data: gz {os.path.basename(path_in_repo)}",
        "content": base64.b64encode(gz).decode(),
        "branch": BRANCH,
    }
    if cur:
        try:
            payload["sha"] = json.loads(cur)["sha"]
        except Exception:
            pass
    code, body = api("PUT", f"{BASE}/contents/{path_in_repo}", payload)
    return code, len(gz)


def main():
    total, ok, fail = 0, 0, 0
    shards = sorted(glob.glob(os.path.join(ROOT, "hf-profiles-shard*")))
    if not shards:
        print("ERROR: artifact 目录为空", flush=True)
        sys.exit(1)
    for d in shards:
        sname = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isfile(f):
                continue
            with open(f, "rb") as fh:
                data = fh.read()
            dest = f"modelscope_output/gz/{sname}/{os.path.basename(f)}.gz"
            code, gzlen = put_gz(dest, data)
            total += 1
            status = "OK" if code in (200, 201) else f"FAIL({code})"
            print(f"{status} {dest} (raw {len(data)/1e6:.1f}MB -> gz {gzlen/1e6:.2f}MB)", flush=True)
            if code in (200, 201):
                ok += 1
            else:
                fail += 1
    print(f"DONE ok={ok} fail={fail} total={total}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
