# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local terminal lifecycle tests; never launch Hermes or a live model."""

from nemo_fabric_adapters.hermes import interfaces
from nemo_fabric_adapters.hermes.terminal import TerminalInterface
import os
import sys


async def test_runtime_publishes_an_executable_terminal_entry_without_secrets(tmp_path):
    terminal = TerminalInterface()
    directory = tmp_path / "interfaces"
    try:
        await terminal.start(
            directory,
            command=[sys.executable, "-c", "pass"],
            environment={**os.environ, "TEST_KEY": "not-in-launcher"},
            workspace=str(tmp_path),
        )
        launcher = directory / "terminal"
        assert launcher.is_file()
        assert os.access(launcher, os.X_OK)
        assert "not-in-launcher" not in launcher.read_text()
        assert (directory / "terminal.running").is_file()
    finally:
        await terminal.stop()
    assert not (directory / "terminal.running").exists()


def test_terminal_only_needs_no_http_port():
    assert interfaces.interface_ports({"interfaces": {"terminal": {}}}) == {}


def test_terminal_does_not_change_http_ports():
    assert interfaces.interface_ports(
        {"interfaces": {"terminal": {}, "api": {"port": 8642}}}
    ) == {"api": 8642}


async def test_stop_waits_for_terminal_exit_and_invalidates_old_launcher(tmp_path):
    import asyncio
    import pty
    import select

    terminal = TerminalInterface()
    directory = tmp_path / "interfaces"
    master, slave = pty.openpty()
    launcher_process = None
    try:
        await terminal.start(
            directory,
            command=[
                sys.executable,
                "-c",
                "import time; print('terminal-ready', flush=True); time.sleep(60)",
            ],
            environment=dict(os.environ),
            workspace=str(tmp_path),
        )
        launcher_process = await asyncio.create_subprocess_exec(
            str(directory / "terminal"),
            stdin=slave,
            stdout=slave,
            stderr=slave,
        )

        def read_ready():
            assert select.select([master], [], [], 3)[0], "terminal did not start"
            return os.read(master, 4096)

        assert b"terminal-ready" in await asyncio.to_thread(read_ready)
        await terminal.stop()
        assert await asyncio.wait_for(launcher_process.wait(), 2) != 0
        rejected = await asyncio.create_subprocess_exec(
            str(directory / "terminal"),
            stdin=slave,
            stdout=slave,
            stderr=slave,
        )
        assert await asyncio.wait_for(rejected.wait(), 2) != 0
    finally:
        await terminal.stop()
        if launcher_process is not None and launcher_process.returncode is None:
            launcher_process.kill()
            await launcher_process.wait()
        os.close(master)
        os.close(slave)


async def test_stop_waits_for_a_child_created_during_launch_cancellation(
    tmp_path, monkeypatch
):
    import array
    import asyncio
    import json
    import pty
    import socket
    import struct

    terminal = TerminalInterface()
    terminal._command = [sys.executable, "-c", "import time; time.sleep(60)"]
    terminal._environment = dict(os.environ)
    terminal._workspace = str(tmp_path)
    terminal._stopping = False
    master, slave = pty.openpty()
    client, server = socket.socketpair()
    spawned = asyncio.Event()
    release = asyncio.Event()
    children = []
    original_spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        child = await original_spawn(*args, **kwargs)
        children.append(child)
        spawned.set()
        await release.wait()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    body = json.dumps([]).encode()
    client.sendmsg(
        [struct.pack("!I", len(body)) + body],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [slave] * 3))],
    )
    serving = asyncio.create_task(terminal._serve(server))
    try:
        await asyncio.wait_for(spawned.wait(), 2)
        serving.cancel()
        release.set()
        await asyncio.wait_for(serving, 3)
        assert children[0].returncode is not None
    finally:
        release.set()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        for child in children:
            if child.returncode is None:
                child.kill()
            await child.wait()
        client.close()
        os.close(master)
        os.close(slave)


async def test_stop_kills_terminal_descendant_when_launcher_exits_first():
    import asyncio
    import signal
    from nemo_fabric_adapters.hermes.terminal import _stop_child

    descendant = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    wrapper = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {descendant!r}]); time.sleep(60)"
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        wrapper,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(child.stdout.readline(), 2) == b"ready\n"
        await asyncio.wait_for(_stop_child(child), 3)
        assert await asyncio.wait_for(child.stdout.read(), 1) == b""
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await child.wait()


def test_terminal_setting_is_opt_in():
    assert not interfaces.terminal_enabled({})
    assert not interfaces.terminal_enabled(
        {"interfaces": {"terminal": {"enabled": False}}}
    )
    assert interfaces.terminal_enabled({"interfaces": {"terminal": {}}})


async def test_terminal_starts_owned_gateway_before_publishing_launcher(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock, MagicMock

    children = interfaces.InterfaceChildren()
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    identity = AsyncMock(
        side_effect=[
            None,
            {"pid": 12345, "hermes_home": str(tmp_path), "state": "ready"},
        ]
    )
    monkeypatch.setattr(interfaces, "_gateway_identity", identity)
    monkeypatch.setattr(children._terminal, "start", AsyncMock())
    await children.start(
        {},
        hermes_home=tmp_path,
        token_path=tmp_path / "api.token",
        timeout=2,
        environment=dict(os.environ),
        terminal=True,
        workspace=str(tmp_path),
    )
    assert spawn.call_args.args[-3:] == ("gateway", "run", "--external-supervisor")
    children._terminal.start.assert_awaited_once()
    assert children._terminal.start.call_args.kwargs["command"] == [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "--tui",
    ]


async def test_terminal_refuses_existing_gateway_without_signalling_it(
    tmp_path, monkeypatch
):
    import pytest
    from unittest.mock import AsyncMock
    from nemo_fabric_adapters.common.lifecycle import LifecycleError

    children = interfaces.InterfaceChildren()
    monkeypatch.setattr(
        interfaces, "_gateway_identity", AsyncMock(return_value={"pid": 99})
    )
    spawn = AsyncMock()
    monkeypatch.setattr(interfaces.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(LifecycleError, match="already owns"):
        await children.start(
            {},
            hermes_home=tmp_path,
            token_path=tmp_path / "api.token",
            timeout=2,
            environment={},
            terminal=True,
            workspace=str(tmp_path),
        )
    spawn.assert_not_awaited()


def test_descriptor_accepts_terminal_without_a_port():
    import json
    from pathlib import Path
    import jsonschema

    descriptor = json.loads(
        Path("adapters/python/hermes/hermes.fabric-adapter.json").read_text()
    )
    jsonschema.validate({"interfaces": {"terminal": {}}}, descriptor["settings_schema"])


async def test_cancelling_stop_still_removes_terminal_socket(tmp_path):
    import asyncio
    import pytest

    terminal = TerminalInterface()
    await terminal.start(
        tmp_path,
        command=[sys.executable, "-c", "pass"],
        environment=dict(os.environ),
        workspace=str(tmp_path),
    )
    terminal._accept_task.cancel()
    await asyncio.gather(terminal._accept_task, return_exceptions=True)
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def accepted_client():
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()

    client = asyncio.create_task(accepted_client())
    terminal._clients.add(client)
    await asyncio.sleep(0)
    stop = asyncio.create_task(terminal.stop())
    await cleaning.wait()
    stop.cancel()
    release.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await stop
        assert terminal._socket_dir is None
    finally:
        await terminal.stop()


async def test_terminal_cleanup_error_does_not_skip_owned_gateway_cleanup(monkeypatch):
    import pytest
    from unittest.mock import AsyncMock, MagicMock

    children = interfaces.InterfaceChildren()
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.wait = AsyncMock(return_value=0)
    children._children = [interfaces._Child("gateway", 0, process)]
    monkeypatch.setattr(
        children._terminal,
        "stop",
        AsyncMock(side_effect=RuntimeError("terminal cleanup failed")),
    )
    signal_group = MagicMock()
    monkeypatch.setattr(interfaces.common_utils, "signal_process_group", signal_group)
    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        await children.stop()
    process.wait.assert_awaited_once()
    assert signal_group.call_count == 2


async def test_terminal_refuses_launch_after_its_gateway_exits(tmp_path, monkeypatch):
    import array
    import asyncio
    import json
    import pty
    import socket
    import struct
    from unittest.mock import AsyncMock

    terminal = TerminalInterface()
    await terminal.start(
        tmp_path,
        command=[sys.executable, "-c", "pass"],
        environment=dict(os.environ),
        workspace=str(tmp_path),
        owner_alive=lambda: False,
    )
    master, slave = pty.openpty()
    client, server = socket.socketpair()
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    try:
        body = json.dumps([]).encode()
        client.sendmsg(
            [struct.pack("!I", len(body)) + body],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [slave] * 3))],
        )
        await terminal._serve(server)
        assert client.recv(1) == b""
        spawn.assert_not_awaited()
    finally:
        await terminal.stop()
        client.close()
        os.close(master)
        os.close(slave)
