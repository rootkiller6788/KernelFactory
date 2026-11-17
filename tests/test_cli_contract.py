"""CLI 契约：`--help`、每个子命令的 `--help`、以及 `check`（若有）都要 rc=0。

⚠️ 自包含：不依赖 conftest 夹具（有的项目自带别的 `conftest.py`）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    # ⚠️ 故意**剥掉** UTF-8 相关环境变量：测试要验证 CLI 自己在 locale 编码
    #    （Windows 上是 GBK）下也能输出 UTF-8，而不是靠调用者预先设好环境。
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


def parse_subcommands(text: str) -> list[str]:
    """从 `--help` 输出里解析子命令名（两种写法都吃）。"""
    subs: set[str] = set()
    m = re.search(r"\{([a-z0-9,\-]+)\}", text)
    if m:
        subs |= set(m.group(1).split(","))
    for line in text.splitlines():
        m2 = re.match(r"^ {4}([a-z][a-z0-9\-]*)\s{2,}\S", line)
        if m2:
            subs.add(m2.group(1))
    return sorted(subs)


def test_top_level_help() -> None:
    r = run_cli("--help")
    assert r.returncode == 0, r.stderr


def test_version_flag() -> None:
    """顶层 `--version`：不需要子命令、不读配置、不跑流水线。"""
    r = run_cli("--version")
    assert r.returncode == 0, r.stderr
    assert re.fullmatch(r"kernelfactory \d+\.\d+\.\d+", r.stdout.strip()), r.stdout


def test_every_subcommand_help() -> None:
    subs = parse_subcommands(run_cli("--help").stdout)
    assert subs, "没从 --help 里解析出任何子命令"
    bad = [f"{s}: rc={r.returncode}"
           for s in subs
           for r in [run_cli(s, "--help")]
           if r.returncode != 0]
    assert not bad, "有子命令的 --help 非零退出：\n" + "\n".join(bad)


def test_check_exit_zero() -> None:
    subs = parse_subcommands(run_cli("--help").stdout)
    if "check" not in subs:
        pytest.skip("本项目没有 check 子命令")
    r = run_cli("check")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def test_help_survives_non_utf8_locale() -> None:
    """`--help` 在 locale 编码（GBK）下也必须 rc=0，且是 UTF-8 可解码的。"""
    r = run_cli("--help")
    assert r.returncode == 0, r.stderr
    assert "会写盘" in r.stdout


@pytest.mark.parametrize("cmd", ["run", "visualize", "check",
                                 "verify", "show", "dump-config"])
def test_each_help_has_exactly_one_write_marker(cmd: str) -> None:
    """A.3 契约的机器侧：每个子命令恰好一个 `会写盘` / `只读` 标记。"""
    out = run_cli(cmd, "--help").stdout
    assert ("会写盘" in out) + ("只读" in out) == 1, out


def test_check_json_is_exactly_one_document() -> None:
    """`--json` 的 stdout 上只许有一段 JSON（可直接 `| jq`）。"""
    r = run_cli("check", "--json")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    doc = json.loads(r.stdout)          # 整个 stdout 必须正好是一段 JSON
    assert doc["cmd"] == "check"
    assert doc["acceptance"]["n"] >= 18
    assert isinstance(doc["in_memory_rerun"], bool)


def test_show_json_is_exactly_one_document() -> None:
    r = run_cli("show", "--json")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    doc = json.loads(r.stdout)
    assert doc["cmd"] == "show"
    assert doc["summary"]["n_ops"] == 3
