"""路径锚定契约：`--config` / `--out` / `--target` 按**项目根**解析，不按 cwd。

README A.3 承诺「从工作区根调用与从项目目录调用写到同一个地方」。
这条承诺很容易在实现里悄悄失效：只要某个调用点漏了 `root=`，
从别的目录调用就会读到另一份配置/产物，而两条路径都不报错 ——
于是「今天恰好对」与「对」无法区分（C.323 的同一母题）。

测试手法：**把项目复制到临时目录，再从另一个目录调用**。这样即使实现
漏了锚定，写出的东西也落在 tmp 里，不会污染仓库。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
OUT = ROOT / "out"
CONFIGS = ROOT / "configs"

pytestmark = pytest.mark.skipif(
    not (OUT / "summary.json").exists(),
    reason="需要先跑 `python main.py run` 生成 out/",
)


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


@pytest.fixture()
def copied_project(tmp_path: Path) -> tuple[Path, Path]:
    """(项目副本, 另一个工作目录)。"""
    proj = tmp_path / "proj"
    shutil.copytree(OUT, proj / "out")
    shutil.copytree(CONFIGS, proj / "configs")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return proj, elsewhere


def test_relative_out_is_anchored_to_root(copied_project) -> None:
    proj, elsewhere = copied_project
    r = run_cli("--root", str(proj), "visualize", "--out", "out", "--json",
                cwd=elsewhere)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    doc = json.loads(r.stdout)
    # 相对路径 `out` 被解析成 **root/out**，不是 cwd/out。
    assert Path(doc["out"]) == proj / "out"
    assert doc["all_identical"] is True


def test_relative_target_is_anchored_to_root(copied_project) -> None:
    proj, elsewhere = copied_project
    r = run_cli("--root", str(proj), "visualize", "--target", "reports",
                cwd=elsewhere)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert (proj / "reports" / "report.md").exists()
    assert (proj / "reports" / "report.html").exists()
    assert not (elsewhere / "reports").exists()


def test_default_root_is_the_project_not_cwd(tmp_path: Path) -> None:
    """不传 `--root` 时也必须用项目根，而不是当前工作目录。"""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # 默认 root = main.py 所在项目，与 cwd 无关。
    r = run_cli("show", "--json", cwd=elsewhere)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    doc = json.loads(r.stdout)
    assert doc["summary"]["n_ops"] == 3
