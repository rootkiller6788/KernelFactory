"""中断/断管：命令行不该把 traceback 当成普通输出。

`python main.py check --json | head -1` 会在下游关闭管道时触发
`BrokenPipeError`；`Ctrl-C` 触发 `KeyboardInterrupt`。两者都是**预期事件**，
不是 bug —— 但未捕获时它们会以 traceback 的形式混进 stderr，
让「命令坏了」和「使用者提前结束」看起来一样。
"""
from __future__ import annotations

from kernelfactory import cli


def _raise(exc: BaseException):
    def _cmd(_args):
        raise exc
    return _cmd


def test_broken_pipe_exits_quietly(monkeypatch, capsys) -> None:
    monkeypatch.setitem(cli._COMMANDS, "show", _raise(BrokenPipeError()))
    assert cli.main(["show"]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_keyboard_interrupt_exits_130(monkeypatch, capsys) -> None:
    monkeypatch.setitem(cli._COMMANDS, "show", _raise(KeyboardInterrupt()))
    assert cli.main(["show"]) == 130
    assert "已中断" in capsys.readouterr().err
