"""Closed-loop HTTP load generator for the detection service.

    python tools/loadtest.py --url http://localhost:8000 --concurrency 1 4 16 64

Each of `concurrency` virtual clients uploads an image, waits for the answer,
and immediately sends the next - the way a fleet of cameras or an upstream
service behaves, and the setup in which Little's law ties throughput,
latency and concurrency together (throughput ≈ concurrency / latency).

The clients are spread over several processes. A single Python process
building multipart bodies and parsing JSON tops out at a few thousand
requests per second; spreading the load keeps the generator from being the
thing that is measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from benchlib import SAMPLE_IMAGE, summarize


async def _client_loop(url, payload, clients, warmup_s, duration_s, endpoint):
    raw = endpoint.endswith("/raw")
    import httpx2 as httpx

    limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
    timeout = httpx.Timeout(30.0)
    records: list[tuple[float, float, int]] = []  # (finished_at, latency_ms, status)
    start = time.perf_counter()
    measure_from = start + warmup_s
    stop_at = measure_from + duration_s

    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=timeout) as http:

        async def worker():
            while True:
                t0 = time.perf_counter()
                if t0 >= stop_at:
                    return
                try:
                    if raw:  # the image is the body; no multipart framing
                        response = await http.post(
                            endpoint, content=payload, headers={"content-type": "image/jpeg"}
                        )
                    else:
                        response = await http.post(
                            endpoint, files={"file": ("image.jpg", payload, "image/jpeg")}
                        )
                    status = response.status_code
                except Exception:
                    status = 0
                t1 = time.perf_counter()
                if t0 >= measure_from and t1 <= stop_at:
                    records.append((t1, (t1 - t0) * 1000, status))

        await asyncio.gather(*(worker() for _ in range(clients)))
    return records


def _process_main(args):
    url, payload, clients, warmup_s, duration_s, endpoint = args
    return asyncio.run(_client_loop(url, payload, clients, warmup_s, duration_s, endpoint))


def run_level(
    url: str,
    payload: bytes,
    concurrency: int,
    duration_s: float = 10.0,
    warmup_s: float = 2.0,
    processes: int = 4,
    endpoint: str = "/detect",
) -> dict:
    """Hold `concurrency` requests in flight for `duration_s`; report what happened."""
    processes = max(1, min(processes, concurrency))
    shares = [
        concurrency // processes + (1 if i < concurrency % processes else 0)
        for i in range(processes)
    ]
    jobs = [(url, payload, n, warmup_s, duration_s, endpoint) for n in shares if n]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_process_main, jobs)

    records = [r for part in parts for r in part]
    ok = [latency for _, latency, status in records if status == 200]
    shed = sum(1 for _, _, status in records if status == 503)
    failed = sum(1 for _, _, status in records if status not in (200, 503))
    result = {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "requests": len(records),
        "ok": len(ok),
        "shed_503": shed,
        "failed": failed,
        "throughput_rps": round(len(ok) / duration_s, 1),
    }
    if ok:
        result["latency_ms"] = summarize(ok)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, default=SAMPLE_IMAGE)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()

    payload = args.image.read_bytes()
    for level in args.concurrency:
        result = run_level(args.url, payload, level, args.duration, args.warmup, args.processes)
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
