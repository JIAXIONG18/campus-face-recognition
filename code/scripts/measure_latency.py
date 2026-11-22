"""通过 HTTP 接口测量端到端响应时间。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def post_image(url: str, path: Path):
    boundary = uuid.uuid4().hex
    body = b""
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"scene\"\r\n\r\n考勤\r\n".encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/api/recognize")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    paths = [Path(p) for p, _ in splits["stage2"]["test"]][: args.n]
    times, server_times = [], []
    for path in paths:
        elapsed, payload = post_image(args.url, path)
        times.append(elapsed)
        server_times.append(payload.get("elapsed_ms", elapsed))
    times.sort()
    summary = {
        "n": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[int(len(times) * 0.95) - 1],
        "server_mean_ms": statistics.mean(server_times),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    metrics_path = ROOT / "experiments" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["e2e_ms"] = summary["mean_ms"]
    metrics["e2e_median_ms"] = summary["median_ms"]
    metrics["e2e_p95_ms"] = summary["p95_ms"]
    metrics_path.write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
