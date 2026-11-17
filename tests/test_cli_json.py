"""`--json` 的输出契约：写当前 stdout、必须是**严格合法**的 JSON。

这两条都不是文案偏好：

- 写当前 stdout：否则 `contextlib.redirect_stdout` / pytest 捕获 / 上层接管道
  都会读到空输出，而命令本身 rc=0 —— 一个「看起来成功了」的假通道；
- `allow_nan=False`：Python 的 `json` 默认允许 `NaN` / `Infinity`，
  但这两者**不是 JSON**。`main.py check --json | jq` 会在下游炸，
  而错误离产生它的地方很远。
"""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from kernelfactory.cli import _emit_json


def test_emit_json_writes_to_current_stdout() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_json({"b": 1, "a": "中文"})
    assert json.loads(buf.getvalue()) == {"a": "中文", "b": 1}


def test_emit_json_is_sorted_and_human_readable() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_json({"z": 1, "a": "⇒"})
    text = buf.getvalue()
    assert text.index('"a"') < text.index('"z"')   # sort_keys=True
    assert "⇒" in text                              # ensure_ascii=False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_emit_json_rejects_non_finite(bad: float) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(ValueError):
        _emit_json({"x": bad})
    assert buf.getvalue() == ""     # 失败时不许写出半段 JSON
