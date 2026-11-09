"""Heavy models in child processes, for clients whose models must stay out of the caller (TensorFlow, PyTorch).

    pool = WorkerPool("libcrate.models.essentia.worker", [model_dir, "300"], processes=4,
                      error=EssentiaError, label="Essentia")
    pool.request("genres", "song.flac")   # runs in a free child process, started on first need
    pool.close()                          # stops them all, which frees the models' memory

The child is a fresh interpreter running `python -m MODULE SOCKET_FD ARGS...`, whose
`__main__` calls serve(): it loads its models, says whether that worked, then answers
each (request, argument) it receives until it receives None.

Up to `processes` children run, each handling one request at a time: requests from
several threads run in parallel, one per child. A child starts on the first request
that needs it and they all stop on close(). Their libraries never return memory
while the process lives, so stopping it is the only way to unload them; the calling
process never imports them. If a child dies (a crash on a bad file), that request
fails and the next one on it starts a new child.
"""

import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection

__all__ = ["WorkerPool", "serve"]


class WorkerPool:
    def __init__(self, module: str, args: list[str], *, processes: int, error: type[Exception], label: str):
        """
        module    -> the worker module, run as `python -m module`; its __main__ calls serve().
        args      -> its command-line arguments after the socket (what to load).
        processes -> how many requests can run at once, each in its own child.
        error     -> raised for a failed request or a child that can't start; label names it in messages.
        """
        self.module, self.args, self.error, self.label = module, [str(a) for a in args], error, label
        self.processes = max(1, processes)
        self.workers: list[_Worker] = []
        self._idle: list[_Worker] = []
        self._available = threading.Condition()

    def request(self, name: str, arg):
        """Have a free child run the request `name` on `arg` and return its result, waiting for one if all
        `processes` are busy."""
        with self._available:
            while not self._idle and len(self.workers) >= self.processes:
                self._available.wait()
            if self._idle:
                worker = self._idle.pop()
            else:
                worker = _Worker(self)
                self.workers.append(worker)
        try:
            return worker.request(name, arg)
        finally:
            with self._available:
                self._idle.append(worker)
                self._available.notify()

    def close(self) -> None:
        """Stop the children. Later requests start new ones."""
        for worker in self.workers:
            worker.stop()


class _Worker:
    """One child process, started on its first request (and again after it dies)."""

    def __init__(self, pool: WorkerPool):
        self.pool = pool
        self._lock = threading.Lock()
        self._process = None
        self._conn = None

    def request(self, name: str, arg):
        with self._lock:
            if self._process is None:
                self._start()
            try:
                self._conn.send((name, arg))
                status, result = self._conn.recv()
            except (EOFError, OSError):
                code = self._stop()
                raise self.pool.error(f"the {self.pool.label} process stopped while working on {arg} "
                                      f"(exit code {code})")
            if status == "error":
                raise self.pool.error(result)
            return result

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def _start(self) -> None:
        # A fresh interpreter rather than multiprocessing: that would re-run the caller's main
        # script in the child unless it's guarded by `if __name__ == "__main__"`.
        ours, theirs = socket.socketpair()
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", self.pool.module, str(theirs.fileno()), *self.pool.args],
                pass_fds=[theirs.fileno()], stdin=subprocess.DEVNULL,
            )
        except OSError as e:
            ours.close()
            raise self.pool.error(f"could not start the {self.pool.label} process: {e}") from e
        finally:
            theirs.close()
        self._conn = Connection(ours.detach())
        try:
            status, error = self._conn.recv()  # wait for the models to load
        except (EOFError, OSError):
            status, error = "error", (f"the {self.pool.label} process stopped while loading the models "
                                      f"(exit code {self._stop()})")
        if status == "error":
            self._stop()
            raise self.pool.error(error)

    def _stop(self) -> int | None:
        if self._process is None:
            return None
        try:
            self._conn.send(None)
        except OSError:
            pass  # already gone
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        code = self._process.returncode
        self._conn.close()
        self._process = self._conn = None
        return code


def serve(conn: Connection, load: Callable[[], object], requests: tuple[str, ...], error: type[Exception],
          label: str) -> None:
    """The child's side: load() the models (an object whose methods named in `requests` answer them),
    send ("ok", None) or ("error", message), then answer each (request, argument) with ("ok", result)
    or ("error", message) until None arrives."""
    try:
        models = load()
    except Exception as e:
        conn.send(("error", f"could not load the {label} models: {type(e).__name__}: {e}"))
        return
    conn.send(("ok", None))
    while (message := conn.recv()) is not None:
        name, arg = message
        try:
            if name not in requests:
                raise error(f"unknown request {name!r}")
            conn.send(("ok", getattr(models, name)(arg)))
        except error as e:
            conn.send(("error", str(e)))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))
