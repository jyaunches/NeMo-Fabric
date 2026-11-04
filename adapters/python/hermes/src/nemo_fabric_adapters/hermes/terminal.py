# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own native TUI children while their terminal belongs to sandbox exec.

The private socket transfers file descriptors, not terminal bytes or credentials.
The adapter launches every child itself; clients cannot register arbitrary PIDs.
"""

from __future__ import annotations

import array
import asyncio
import contextlib
import json
import os
import shlex
import signal
import socket
import struct
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

MAX_REQUEST = 8192
STOP_GRACE_SECONDS = 2


def _receive_launch(peer: socket.socket) -> tuple[list[str], list[int]]:
    descriptors = array.array("i")
    try:
        data, ancillary, flags, _ = peer.recvmsg(
            MAX_REQUEST + 4, socket.CMSG_SPACE(3 * descriptors.itemsize)
        )
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                descriptors.frombytes(
                    value[: len(value) - len(value) % descriptors.itemsize]
                )
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(descriptors) != 3:
            raise ValueError("expected three terminal descriptors")
        while len(data) < 4:
            chunk = peer.recv(4 - len(data))
            if not chunk:
                raise ValueError("incomplete launch")
            data += chunk
        length = struct.unpack("!I", data[:4])[0]
        if length > MAX_REQUEST:
            raise ValueError("launch arguments too large")
        while len(data) < length + 4:
            chunk = peer.recv(length + 4 - len(data))
            if not chunk:
                raise ValueError("incomplete launch")
            data += chunk
        arguments = json.loads(data[4:])
        if not isinstance(arguments, list) or not all(
            isinstance(arg, str) and "\0" not in arg for arg in arguments
        ):
            raise ValueError("invalid terminal arguments")
        if not all(os.isatty(fd) for fd in descriptors):
            raise ValueError("terminal launch requires a TTY")
        return arguments, list(descriptors)
    except BaseException:
        for fd in descriptors:
            os.close(fd)
        raise


async def _stop_child(process: asyncio.subprocess.Process) -> None:
    from nemo_fabric_adapters.common.utils import signal_process_group

    signal_process_group(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), STOP_GRACE_SECONDS)
    except TimeoutError:
        pass
    finally:
        # The Python launcher can exit before its Node child. Match the
        # interface supervisor's cleanup of the whole owned process group.
        signal_process_group(process.pid, signal.SIGKILL)
    await process.wait()


class TerminalInterface:
    """One runtime's terminal admission and owned children."""

    def __init__(self) -> None:
        self._server: socket.socket | None = None
        self._accept_task: asyncio.Task | None = None
        self._clients: set[asyncio.Task] = set()
        self._socket_dir: tempfile.TemporaryDirectory | None = None
        self._marker: Path | None = None
        self._stopping = True
        self._stop_task: asyncio.Task | None = None
        self._errors: list[BaseException] = []
        self._owner_alive: Callable[[], bool] | None = None

    async def start(
        self,
        directory: Path,
        *,
        command: list[str],
        environment: dict[str, str],
        workspace: str,
        owner_alive: Callable[[], bool] | None = None,
    ) -> None:
        self._command = command
        self._stop_task = None
        self._errors.clear()
        self._environment = environment.copy()
        self._workspace = workspace
        self._owner_alive = owner_alive
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # AF_UNIX paths are short even when the artifact root is deeply nested.
        # A new private directory also fences old launchers after runtime recovery.
        self._socket_dir = tempfile.TemporaryDirectory(
            prefix="fabric-hermes-", dir="/tmp"
        )
        address = str(Path(self._socket_dir.name) / "terminal.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        try:
            server.bind(address)
            server.listen()
            server.setblocking(False)
            self._stopping = False
            self._accept_task = asyncio.create_task(self._accept())
            # The forwarding client uses only stdlib. Avoid loading installed
            # harnesses or editable-package hooks on each terminal connection.
            invocation = shlex.join(
                [sys.executable, "-S", str(Path(__file__).resolve()), address]
            )
            launcher = directory / "terminal"
            launcher.write_text(
                f'#!/bin/sh\nexec {invocation} "$@"\n', encoding="utf-8"
            )
            launcher.chmod(0o700)
            self._marker = directory / "terminal.running"
            self._marker.write_text(address + "\n", encoding="utf-8")
            self._marker.chmod(0o600)
        except BaseException:
            await self.stop()
            raise

    async def _accept(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            peer, _ = await loop.sock_accept(self._server)
            task = asyncio.create_task(self._serve(peer))
            self._clients.add(task)
            task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task) -> None:
        self._clients.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._errors.append(error)

    async def _serve(self, peer: socket.socket) -> None:
        process = None
        descriptors: list[int] = []
        receive = None
        received = False
        spawn = None
        controls = None
        wait = None
        try:
            peer.settimeout(2)
            receive = asyncio.create_task(asyncio.to_thread(_receive_launch, peer))
            arguments, descriptors = await asyncio.shield(receive)
            received = True
            if self._stopping or (
                self._owner_alive is not None and not self._owner_alive()
            ):
                return
            # No shell, arbitrary PID, environment override or executable supplied
            # by the client. Native Hermes owns parsing its ordinary CLI arguments.
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *self._command,
                    *arguments,
                    cwd=self._workspace,
                    env=self._environment,
                    stdin=descriptors[0],
                    stdout=descriptors[1],
                    stderr=descriptors[2],
                    start_new_session=True,
                )
            )
            process = await asyncio.shield(spawn)
            for fd in descriptors:
                os.close(fd)
            descriptors = []
            peer.setblocking(False)
            loop = asyncio.get_running_loop()
            await loop.sock_sendall(peer, b"R")
            controls = asyncio.create_task(self._controls(peer, process))
            wait = asyncio.create_task(process.wait())
            done, _ = await asyncio.wait(
                {controls, wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if wait not in done:
                await _stop_child(process)
            await loop.sock_sendall(peer, struct.pack("!i", process.returncode))
        except (OSError, ValueError, asyncio.CancelledError):
            pass
        finally:
            # Cancellation can arrive after the OS created the child but before
            # asyncio returned its handle. Acquire that handle before cleanup.
            if spawn is not None and process is None:
                try:
                    process = await spawn
                except OSError:
                    pass
            # A cancelled recvmsg thread must finish before its descriptors can
            # be reclaimed. Closing the peer wakes it without waiting its timeout.
            if receive is not None and not received:
                with contextlib.suppress(OSError):
                    peer.shutdown(socket.SHUT_RDWR)
                try:
                    _, descriptors = await receive
                except (OSError, ValueError):
                    pass
            for fd in descriptors:
                os.close(fd)
            for task in (controls, wait):
                if task is not None:
                    task.cancel()
            if process is not None:
                await _stop_child(process)
            peer.close()
            await asyncio.gather(
                *(task for task in (controls, wait) if task is not None),
                return_exceptions=True,
            )

    async def _controls(
        self, peer: socket.socket, process: asyncio.subprocess.Process
    ) -> None:
        from nemo_fabric_adapters.common.utils import signal_process_group

        loop = asyncio.get_running_loop()
        while data := await loop.sock_recv(peer, 32):
            for value in data:
                signum = {
                    ord("w"): signal.SIGWINCH,
                    ord("i"): signal.SIGINT,
                    ord("t"): signal.SIGTERM,
                }.get(value)
                if signum is None:
                    return
                if process.returncode is None:
                    signal_process_group(process.pid, signum)

    async def stop(self) -> None:
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await self._stop_task
            raise

    async def _stop(self) -> None:
        self._stopping = True
        if self._marker is not None:
            self._marker.unlink(missing_ok=True)
        if self._accept_task is not None:
            self._accept_task.cancel()
            await asyncio.gather(self._accept_task, return_exceptions=True)
            self._accept_task = None
        if self._server is not None:
            self._server.close()
            self._server = None
        clients = list(self._clients)
        for task in clients:
            task.cancel()
        results = await asyncio.gather(*clients, return_exceptions=True)
        if self._socket_dir is not None:
            self._socket_dir.cleanup()
            self._socket_dir = None
        for result in results:
            if isinstance(result, Exception):
                raise result
        if self._errors:
            raise self._errors[0]


def main() -> int:
    """Transfer this terminal to the runtime and wait for its child's exit."""
    if len(sys.argv) < 2:
        return 2
    arguments = json.dumps(sys.argv[2:]).encode()
    if len(arguments) > MAX_REQUEST:
        return 2
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous = {}
    try:
        peer.connect(sys.argv[1])
        packet = struct.pack("!I", len(arguments)) + arguments
        sent = peer.sendmsg(
            [packet],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [0, 1, 2]))],
        )
        peer.sendall(packet[sent:])
        if peer.recv(1) != b"R":
            raise OSError("terminal launch refused")
        for signum, value in (
            (signal.SIGWINCH, b"w"),
            (signal.SIGINT, b"i"),
            (signal.SIGTERM, b"t"),
        ):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _s, _f, value=value: peer.sendall(value))
        result = b""
        while len(result) < 4:
            chunk = peer.recv(4 - len(result))
            if not chunk:
                return 1
            result += chunk
        code = struct.unpack("!i", result)[0]
        return code if code >= 0 else 128 - code
    except OSError:
        print(
            "This Fabric terminal runtime is unavailable. Read the current entry point from apply.",
            file=sys.stderr,
        )
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        peer.close()


if __name__ == "__main__":
    raise SystemExit(main())
