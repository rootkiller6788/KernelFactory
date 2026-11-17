"""冒烟：`src/kernelfactory/` 下每个模块（**含子包**）都要能干净 import。

⚠️ 这不是「判据」，是**语法/接线闸**：能 import 不等于对，但 import 不了就一定错。
⚠️ 模块名按**相对路径**拼（`search/bayesian.py` → `kernelfactory.search.bayesian`）——
直接取 `stem` 会把子包模块导成 `kernelfactory.bayesian`，那是**假红**。
⚠️ 自包含：不依赖 conftest 夹具。
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

PKG = "kernelfactory"
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _modules() -> list[str]:
    names: list[str] = []
    for p in sorted((SRC / PKG).rglob("*.py")):
        parts = list(p.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.append(".".join(parts))
    return sorted(set(names))


def test_package_importable() -> None:
    importlib.import_module(PKG)


def test_every_module_imports() -> None:
    names = _modules()
    assert names, "没在 src/kernelfactory/ 下找到任何模块"
    bad: list[str] = []
    for n in names:
        try:
            importlib.import_module(n)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{n}: {type(e).__name__}: {e}")
    assert not bad, "有模块 import 失败：\n" + "\n".join(bad)


def test_python_dash_m_entrypoint() -> None:
    """`python -m kernelfactory --help` 可用（与 main.py 同一条 CLI）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    # 让子进程使用 src/ 下的包，而不是已安装的旧版本（若有）。
    for key in ("PYTHONIOENCODING", "PYTHONUTF8"):
        env.pop(key, None)
    r = subprocess.run(
        [sys.executable, "-m", "kernelfactory", "--help"],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, r.stderr
    assert "kernelfactory" in r.stdout
