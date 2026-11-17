"""落盘：规范化 JSON、内容寻址版本号、产物清单。

## 规范化 JSON 是「逐位相同」的前提

`json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)` + 末尾换行。
不排序键，两次运行只要字典插入顺序不同就会得到不同字节 ——
而插入顺序在 Python 里是**实现细节**，不该渗进产物。
本项目里这条尤其容易踩：`prune_reasons` / `gene_cardinality` / `cells`
都是按 dict 建的，而它们的键来自 `set` 迭代。

## `created_at` 只许出现在 `version.json`

印在 `report.md` 正文里，两份相同输入的报告就不逐位相同了。
所以时间戳只有一个出口，就是版本文件 —— 而版本文件本身不进摘要。

## 版本号只覆盖**确定性产物**

`VERSIONED_FILES` 在 `pipeline.py` 里显式列出，排除耗时、视图、自指三类。
于是同一份输入跑两次：**只有 `version.json`（时间戳）与 `benchmark.json`（耗时）
不同，其余 N 个结论性产物逐位相同**。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    text = canonical_json(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest_text(text)


def content_version(products: dict[str, Any], versioned: Sequence[str]) -> str:
    """`v1-<digest 前 12 位>`。**按固定顺序**拼，顺序本身是契约的一部分。"""
    h = hashlib.sha256()
    for name in versioned:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
        h.update(canonical_json(products[name]).encode("utf-8"))
    return "v1-" + h.hexdigest()[:12]


def emit(out_dir: Path, products: dict[str, Any],
         versioned: Sequence[str], unversioned: Sequence[str],
         extra: Callable[[dict[str, Any]], dict[str, str]] | None = None
         ) -> dict[str, Any]:
    """把全部产物写进 `out_dir`，返回清单（含每个文件的摘要）。

    ## 顺序是**有依赖的**，不能随便排

    版本号必须**先**算出来（它只覆盖确定性产物），然后塞进 `products`
    —— 因为报告要印版本号，而报告本身**不进摘要**（它是视图）。
    于是顺序是：算版本 → 写确定性产物 → 渲染报告 → 记摘要 → 写 version.json。

    反过来（先渲染再算版本）会让报告里的版本号变成上一轮的，
    而它看起来完全正常 —— 只有把两次 run 的产物逐位比才会发现。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version = content_version(products, versioned)
    version_path = out_dir / "version.json"

    manifest: dict[str, Any] = {
        "version": version,
        "versioned_files": list(versioned),
        "unversioned_files": [n for n in list(unversioned) if n != "version.json"],
        "digests": {},
        "created_at": None,
        "note": "created_at 只出现在这里；版本号只覆盖 versioned_files。",
    }
    products["version.json"] = manifest          # 渲染器要读版本号

    digests: dict[str, str] = {}
    for name in list(versioned) + list(unversioned):
        if name == "version.json" or name not in products:
            continue
        digests[name] = write_json(out_dir / name, products[name])

    texts = extra(products) if extra is not None else {}
    for name, text in texts.items():
        p = out_dir / name
        p.write_text(text, encoding="utf-8", newline="\n")
        digests[name] = digest_text(text)
    if texts:
        manifest["unversioned_files"] = list(manifest["unversioned_files"]) + \
            [n for n in texts if n not in manifest["unversioned_files"]]

    manifest["digests"] = {k: digests[k] for k in sorted(digests)}
    manifest["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(version_path, manifest)
    return manifest


def read_products(out_dir: Path, names: Sequence[str]) -> dict[str, Any]:
    """从 `out/` 读回产物（供只读命令 `visualize` 使用）。**一行都不重跑流水线。**"""
    out: dict[str, Any] = {}
    for name in names:
        p = Path(out_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"产物缺失：{p}")
        out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


__all__ = ["canonical_json", "digest_text", "write_json", "content_version",
           "emit", "read_products"]
