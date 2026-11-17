"""退出码契约：**命令的结论必须能从 rc 读出来**。

两个真实会骗人的场景：

- `run` 把验收失败写进产物、打印在屏幕上，却返回 0 —— CI 里一切「通过」；
- `visualize` 报告「不一致」，却返回 0 —— 一个专门用来比对的命令，
  比对失败不作数，那它就只是个渲染命令。

判据分开测：纯函数测映射，集成测试测 `visualize` 真的把不一致变成 rc=1。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernelfactory.cli import _acceptance_failed, _mismatched_reports

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
OUT = ROOT / "out"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def test_acceptance_failed_maps_gates_and_cases() -> None:
    assert not _acceptance_failed({"failed": [], "n_cases_failed": None})
    assert not _acceptance_failed({"failed": [], "n_cases_failed": 0})
    assert _acceptance_failed({"failed": ["min_ops"], "n_cases_failed": 0})
    assert _acceptance_failed({"failed": [], "n_cases_failed": 1})


def test_mismatched_reports_ignores_missing_files() -> None:
    rows = [
        {"file": "a", "exists": True, "identical": True},
        {"file": "b", "exists": True, "identical": False},
        {"file": "c", "exists": False, "identical": False},
    ]
    assert _mismatched_reports(rows) == ["b"]


@pytest.mark.skipif(not (OUT / "summary.json").exists(),
                    reason="需要先跑 `python main.py run` 生成 out/")
def test_visualize_clean_copy_is_zero(tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(OUT, out)
    r = run_cli("visualize", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "逐位相同" in r.stdout


@pytest.mark.skipif(not (OUT / "summary.json").exists(),
                    reason="需要先跑 `python main.py run` 生成 out/")
def test_corrupt_product_is_a_clear_error(tmp_path: Path) -> None:
    """坏产物必须报「哪个文件坏了」并 rc=2，而不是 JSONDecodeError traceback。"""
    out = tmp_path / "out"
    shutil.copytree(OUT, out)
    (out / "summary.json").write_text("{ 这不是 JSON", encoding="utf-8")
    r = run_cli("visualize", "--out", str(out))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "产物错误" in r.stderr
    assert "summary.json" in r.stderr
    assert "Traceback" not in r.stderr


@pytest.mark.skipif(not (OUT / "summary.json").exists(),
                    reason="需要先跑 `python main.py run` 生成 out/")
def test_visualize_mismatch_is_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(OUT, out)
    report = out / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n陈旧的一行\n",
                      encoding="utf-8", newline="\n")
    r = run_cli("visualize", "--out", str(out))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "不一致" in r.stdout
