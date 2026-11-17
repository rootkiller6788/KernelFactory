"""只读重建：从 `out/` 的产物重造报告，**一行都不重跑流水线**。

## 为什么这条命令值得单独存在

它是「渲染器只吃产物字典」这条纪律的**验收手段**。若哪天有人在
`render_html` 里读了 `run.evaluators`，这条命令会立刻炸 —— 而它炸得很早、
很便宜（毫秒级），比等到某天想复用报告时才发现要好得多。

## 只读就是只读

默认**不写盘**。它的输出是「现有报告与重建结果是否逐位相同」，
而不是「我把报告重写了一遍」。只读命令跑完 `out/` 必须逐位不变 ——
否则「跑完一遍 out/ 没变」这条核对就没有意义了。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .emit import read_products
from .pipeline import UNVERSIONED_FILES, VERSIONED_FILES
from .report import render_html, render_markdown

#: 重建报告所需的产物（= 全部确定性产物 + 耗时 + 版本）。
REQUIRED: tuple[str, ...] = tuple(VERSIONED_FILES) + ("benchmark.json", "version.json")


def rebuild(out_dir: Path) -> dict[str, str]:
    """从 `out/` 读出产物并重造两份报告。**不写盘。**"""
    products = read_products(Path(out_dir), REQUIRED)
    return {"report.md": render_markdown(products),
            "report.html": render_html(products)}


def compare(out_dir: Path) -> dict[str, Any]:
    """重建结果与现有报告逐位比。**只读。**"""
    out_dir = Path(out_dir)
    texts = rebuild(out_dir)
    rows: list[dict[str, Any]] = []
    for name, text in texts.items():
        p = out_dir / name
        existing = p.read_text(encoding="utf-8") if p.exists() else None
        rows.append({
            "file": name,
            "exists": existing is not None,
            "identical": existing == text,
            "existing_bytes": len(existing.encode("utf-8")) if existing else 0,
            "rebuilt_bytes": len(text.encode("utf-8")),
        })
    return {
        "out_dir": str(out_dir),
        "files": rows,
        "all_identical": all(r["identical"] for r in rows),
        "note": "只读：本命令不写盘。",
    }


def render_to(out_dir: Path, target: Path) -> dict[str, Any]:
    """把重建的报告写到别处（**不是 `out/`**）—— 唯一会落盘的路径，且需显式指定。"""
    texts = rebuild(out_dir)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in texts.items():
        (target / name).write_text(text, encoding="utf-8", newline="\n")
        written.append(str(target / name))
    return {"written": written, "note": "写到了 out/ 之外的目标目录。"}


__all__ = ["REQUIRED", "rebuild", "compare", "render_to", "UNVERSIONED_FILES"]
