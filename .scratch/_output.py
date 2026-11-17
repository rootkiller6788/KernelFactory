"""证据脚本共用的输出配置。

`.scratch/` 下的脚本会打印 `✅` / `❌` / `⚠`；Windows 上 stdout 接管道时
默认是 locale 编码（GBK），这些字符会直接触发 `UnicodeEncodeError` ——
于是「核对脚本」自己先崩了，看起来像被核对的对象有问题。
"""
from __future__ import annotations

import sys


def configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
