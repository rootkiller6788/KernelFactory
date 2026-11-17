"""只读命令的契约：跑完之后 `out/` 与 `configs/` **逐位不变**。

为什么不能只看「命令说自己只读」：`dump-config` 名字像 dump，实际写
`configs/default.json`；而它写回的是同一份内容，所以任何 md5 比对都发现不了
（README 附录 C.323）。这里把范围定成**命令真正能写到的地方**（out + configs），
并把「只读」与「写盘」分开测 —— 本文件只覆盖只读的三条。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
OUT = ROOT / "out"
CONFIGS = ROOT / "configs"


def snapshot() -> dict[str, str]:
    files: dict[str, str] = {}
    for base in (OUT, CONFIGS):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = \
                    hashlib.sha256(p.read_bytes()).hexdigest()
    return files


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


@pytest.mark.skipif(not (OUT / "summary.json").exists(),
                    reason="需要先跑 `python main.py run` 生成 out/")
@pytest.mark.parametrize("cmd", ["check", "visualize", "show"])
def test_readonly_command_leaves_products_untouched(cmd: str) -> None:
    before = snapshot()
    r = run_cli(cmd)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    after = snapshot()
    assert after == before, (
        f"`{cmd}` 改动了文件："
        f"{sorted(set(before) ^ set(after) | {k for k in before if after.get(k) != before[k]})}"
    )


@pytest.mark.skipif(not (OUT / "summary.json").exists(),
                    reason="需要先跑 `python main.py run` 生成 out/")
def test_visualize_reports_itself_read_only() -> None:
    r = run_cli("visualize", "--json")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    doc = json.loads(r.stdout)
    assert doc["read_only"] is True
    assert doc["written"] == []
    assert doc["all_identical"] is True
