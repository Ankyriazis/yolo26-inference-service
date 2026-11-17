"""Dynamic batching: coalesce concurrent requests into one forward pass.

A GPU runs a batch of eight images in far less than eight times the time of
one, but HTTP delivers images one at a time. The batcher closes that gap:
requests wait in a queue, and a single inference thread takes everything that
has arrived - up to `max_batch_size`, or until the oldest request has waited
`max_batch_wait_ms` - and runs it as one batch.

Under light load a request finds the queue empty, waits at most the batching
window, and runs alone. Under heavy load requests pile up while the previous
batch is on the GPU, so the next batch is already full and nobody waits for
the window at all. The batch size adapts to the load with no tuning.

Two properties fall out of the design rather than being bolted on:

- The event loop never blocks on inference. All model work happens on one
  dedicated thread, so health checks and uploads stay responsive while the
  GPU is busy.
- The queue is bounded. When it is full the service refuses work immediately
  (`QueueFullError` → HTTP 503) instead of letting latency grow without limit.
  A fast "no" lets a load balancer try another replica; a slow "yes" that
  arrives after the client's timeout helps no one.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _fine_timers(enable: bool) -> None:
    """Ask Windows for 1 ms timer resolution while the batcher runs.

    Windows rounds event-loop timeouts up to the system timer tick, 15.6 ms by
    default, which silently turns a 2 ms batching window into a 16 ms one -
    measured here as a 27 ms request whose batch took 6 ms. Since Windows 10
    2004 the setting is per process. A no-op on every other platform, whose
    timers are already fine-grained.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(OSError, AttributeError):
        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)


class QueueFullError(RuntimeError):
    """The batching queue is at capacity; the caller should shed the request."""


class BatcherClosedError(RuntimeError):
    """The batcher has been stopped and accepts no more work."""


@dataclass
class _Pending[In, Out]:
    item: In
    future: asyncio.Future[Out]
    enqueued: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class BatchReport:
    """What happened to one batch. Passed to the `on_batch` observer."""

    size: int
    queue_wait_ms: list[float]
    inference_ms: float


class DynamicBatcher[In, Out]:
    """Collects items submitted from async code and runs them in batches.

    Args:
        infer: called on the inference thread with a list of inputs; must
            return one output per input, in order. An output may be an
            exception instance, which fails that one caller only.
        max_batch_size: largest batch handed to `infer`.
        max_wait_ms: longest the oldest queued item waits for the batch to
            fill. Zero disables waiting: each batch is whatever is queued.
        max_queue: items allowed to wait; `submit` raises `QueueFullError`
            beyond this.
        on_batch: optional observer, called on the event loop after each batch.
    """

    def __init__(
        self,
        infer: Callable[[Sequence[In]], Sequence[Out]],
        max_batch_size: int = 8,
        max_wait_ms: float = 2.0,
        max_queue: int = 256,
        on_batch: Callable[[BatchReport], Any] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._infer = infer
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.max_queue = max_queue
        self._on_batch = on_batch
        self._queue: asyncio.Queue[_Pending[In, Out]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
        # Exactly one worker: the model is not re-entrant, and two threads
        # contending for one GPU only lengthen the tail.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._closed = False
        if self.max_wait > 0:
            _fine_timers(True)
        self._task = asyncio.create_task(self._run(), name="dynamic-batcher")

    async def stop(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._queue is not None:
            while not self._queue.empty():
                pending = self._queue.get_nowait()
                if not pending.future.done():
                    pending.future.set_exception(BatcherClosedError("shutting down"))
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            if self.max_wait > 0:
                _fine_timers(False)

    # ------------------------------------------------------------ public API

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def submit(self, item: In) -> Out:
        """Queue one item and wait for its result."""
        if self._closed or self._queue is None:
            raise BatcherClosedError("the batcher is not running")
        if self._queue.qsize() >= self.max_queue:
            raise QueueFullError(f"{self._queue.qsize()} requests already waiting for the GPU")
        future: asyncio.Future[Out] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Pending(item, future))
        return await future

    # ------------------------------------------------------------ internals

    async def _collect(self) -> list[_Pending[In, Out]]:
        assert self._queue is not None
        first = await self._queue.get()
        batch = [first]
        deadline = first.enqueued + self.max_wait
        while len(batch) < self.max_batch_size:
            # Anything already queued joins immediately, whatever the window.
            if not self._queue.empty():
                batch.append(self._queue.get_nowait())
                continue
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            # Callers that gave up (client disconnect, timeout) are dropped
            # here rather than spending GPU time on an answer nobody reads.
            live = [p for p in batch if not p.future.done()]
            if not live:
                continue

            started = time.perf_counter()
            waits = [(started - p.enqueued) * 1000 for p in live]
            try:
                outputs = await loop.run_in_executor(
                    self._executor, self._infer, [p.item for p in live]
                )
                if len(outputs) != len(live):
                    raise RuntimeError(
                        f"infer returned {len(outputs)} results for {len(live)} inputs"
                    )
            except Exception as error:  # surfaced to every caller in the batch
                logger.exception("batch of %d failed", len(live))
                for pending in live:
                    if not pending.future.done():
                        pending.future.set_exception(error)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000
            for pending, output in zip(live, outputs, strict=True):
                if pending.future.done():
                    continue
                # `infer` may return an exception in place of one result: that
                # input failed (a corrupt image, say) but its neighbours did not.
                if isinstance(output, BaseException):
                    pending.future.set_exception(output)
                else:
                    pending.future.set_result(output)
            if self._on_batch is not None:
                try:
                    self._on_batch(BatchReport(len(live), waits, elapsed_ms))
                except Exception:  # an observer must never take down inference
                    logger.exception("on_batch observer failed")
