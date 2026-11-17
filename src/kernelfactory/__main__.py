"""`python -m kernelfactory <cmd>` 入口 —— 与 `python main.py <cmd>` 等价。

需要包在 `sys.path` 上（`pip install -e .` 或 `PYTHONPATH=src`）；
仓库根目录下的 `python main.py` 不需要安装，两者走同一个 `cli.main`。
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
