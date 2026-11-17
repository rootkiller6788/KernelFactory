"""命令行。

## 路径锚定契约

`--config` / `--out` / `--root` 的**相对路径按项目根解析**，不按当前工作目录。
`_anchor_args` 在 `main()` 里紧跟 `parse_args` 锚一次 ——
不锚的话，「从工作区根调用」与「从项目目录调用」会写到两个不同的地方，
而两者都不报错。

判据不是「调用点写没写 `root=`」，而是**「这个参数有没有参与路径解析」**。
本 CLI 里参与路径解析的有三个：`--config`（→ `load_config(root=...)`）、
`--out`（→ 产物目录）、`--target`（→ `visualize` 的落盘目录）。
**同一 CLI 里三个开关的锚定约定必须一致。**

## `--json` 的纯度

`--json` 时 stdout 上**只许有一段 JSON**（不许先打印说明再追加 JSON，那样没法 `| jq`）。
手法：`_emit_json` 走真正的 stdout，命令里的 `print` 被 `_quiet_prose` 上下文吞掉。

## 只读命令遇缺失产物

在内存里重跑一遍，**绝不落盘**，并明说「本次是内存重跑，未写盘」。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import pipeline
from .config import ConfigError, dump_config, load_config
from .emit import emit
from .report import render_html, render_markdown

ROOT = Path(__file__).resolve().parents[2]     # L6-kernelfactory/


def _configure_output() -> None:
    """把 stdout/stderr 统一成 UTF-8 —— **否则中文输出在 Windows 管道上会崩**。

    Windows 上 `sys.stdout` 接管道/文件时默认用 locale 编码（本机是 GBK），
    而本项目的帮助文本与 JSON 里有 `⚠` / `⇒` 这类字符 ⇒
    `python main.py --help | ...` 与 `main.py check --json > x.json` 会抛
    `UnicodeEncodeError`，且错误发生在 argparse 打印帮助时 —— 看起来像命令本身坏了。

    `errors="replace"` 是第二道保险：即使某个奇怪终端真的无法输出 UTF-8，
    也只会得到一个替换字符，而不是把整条命令打挂。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:          # 例如被替换成 StringIO
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):    # 已被分离/关闭的流
            pass


@contextlib.contextmanager
def _quiet_prose(enabled: bool) -> Iterator[None]:
    """`--json` 时把命令里的 `print` 吞掉，保证 stdout 上只有一段 JSON。"""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _emit_json(obj: Any) -> None:
    """写到**当前的** `sys.stdout`。

    ⚠️ 不能在 import 时把 `sys.stdout` 存成模块常量：那样 `redirect_stdout` /
    pytest 的输出捕获 / 上层重新绑定 stdout 都会失效 —— 而失效方式很隐蔽：
    JSON 跑到旧流里，测试看到空 stdout。流在调用时取，才是它该被读的时刻。
    """
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def _anchor_args(args: argparse.Namespace) -> argparse.Namespace:
    """把相对路径锚到项目根。**紧跟 parse_args 调一次。**"""
    root = Path(getattr(args, "root", None) or ROOT).resolve()
    for name in ("config", "out", "target"):
        v = getattr(args, name, None)
        if v is None:
            continue
        p = Path(v)
        setattr(args, name, p if p.is_absolute() else (root / p))
    args.root = root
    return args


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernelfactory",
        description="KernelFactory：Kernel 生成的搜索闭环（硬约束 = 一致性 + 声明）",
    )
    p.add_argument("--root", default=None, help="项目根（默认 = 本文件所在项目）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        # ⚠️ 默认值必须用 SUPPRESS：否则子解析器会把主解析器已设好的
        #    `--json` 覆写回 False（子解析器的默认值优先级更高）。
        sp.add_argument("--config", default=argparse.SUPPRESS, help="配置 JSON")
        sp.add_argument("--out", default=argparse.SUPPRESS, help="产物目录")
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="stdout 上只输出一段 JSON")

    # ⚠️ `help=` 只出现在**父级列表**里；子命令自己的 `--help` 只印 usage + options。
    #    `description=` 补上后，`main.py <cmd> --help` 才会印出那句话 ——
    #    于是「这个命令写不写盘」这件事**在两条路径上都能读到**
    #    （`.scratch/check_readme.py` 的 CLI 契约核对就靠它）。
    def add(name: str, desc: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=desc, description=desc)
        common(sp)
        return sp

    # ⚠️ 每个 `desc` 里必须**恰好出现一个**显式标记：`会写盘` 或 `只读`。
    #    这不是文案偏好 —— README 附录 A.3 的「写盘」列用的是**同一套标记**，
    #    `.scratch/check_readme.py` 就按这两个词逐命令机器核（C.323）。
    run = add("run", "跑完整流水线（会写盘：写 out/）")

    vis = add("visualize", "从 out/ 重建报告并逐位比对（只读；--target 时才另写他处）")
    vis.add_argument("--target", default=argparse.SUPPRESS,
                     help="需要落盘时写到这个目录（默认不写盘）")

    add("check", "从 out/ 重跑验收与对抗用例（只读）")
    add("verify", "确定性与产物充分性核对（只读：只写临时目录）")
    add("show", "打印摘要（只读）")
    add("dump-config",
        "⚠️ 会写盘：把当前生效的配置写回 configs/default.json")

    return p


# --------------------------------------------------------------------- 子命令
def _acceptance_failed(acc: dict[str, Any]) -> bool:
    """验收门槛或对抗用例任一失败 ⇒ 进程必须用非零 rc 报出去。

    ⚠️ 「打印了失败」不等于「命令失败」。`run` 若不把结论映射到 rc，
    CI/脚本里的 `python main.py run` 在验收全红时仍然 rc=0 ——
    产物里写得很清楚，而调用方看不见。
    """
    return bool(acc.get("failed")) or bool(acc.get("n_cases_failed"))


def _mismatched_reports(rows: list[dict[str, Any]]) -> list[str]:
    """只把「**存在且不一致**」的视图算作不一致。

    产物缺失时 `_products_for_readonly` 会在内存里重跑并在 note 里明说；
    那属于「没有可比对的对象」，不是「比对失败」。
    """
    return [r["file"] for r in rows if r["exists"] and not r["identical"]]


def _resolve(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg = load_config(getattr(args, "config", None), root=args.root)
    out = Path(getattr(args, "out", None) or (args.root / "out"))
    return cfg, out


def cmd_run(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        run = pipeline.build(cfg)
        pipeline.finish(run)
        manifest = emit(
            out, run.products, pipeline.VERSIONED_FILES, pipeline.UNVERSIONED_FILES,
            extra=lambda p: {"report.md": render_markdown(p),
                             "report.html": render_html(p)},
        )
        summary = run.products["summary.json"]
        acc = run.products["acceptance.json"]
        exp = run.products["expectations.json"]
        text = (f"版本 {manifest['version']}\n"
                f"算子 {summary['n_ops']} 个，空间总计 "
                f"{summary['space_size_total']} 个 kernel"
                f"（{summary['space_size_per_op']}）\n"
                f"τ=0 前沿点数 "
                f"{summary['frontier_points_distinct_at_tau0']}\n"
                f"门 1 放行、门 2 拒掉："
                f"{summary['n_gate1_only_anywhere']} 个非法 kernel\n"
                f"头条漂移比（假装 fp32 ÷ 诚实 bf16）："
                f"{summary['headline_drift_ratio']}\n"
                f"期望：{summary['expectations']['n_held']} 成立 / "
                f"{summary['expectations']['n_refuted']} 被推翻 / "
                f"{exp['n_unmeasurable']} 算不出来\n"
                f"门槛：{summary['acceptance']['n_passed']}/"
                f"{summary['acceptance']['n']} 通过\n"
                f"对抗用例："
                f"{summary['acceptance']['n_cases'] - (acc['n_cases_failed'] or 0)}/"
                f"{summary['acceptance']['n_cases']} 成立\n"
                f"产物目录：{out}")
        if not json_mode:
            print(text)
    if json_mode:
        _emit_json({"cmd": "run", "out": str(out),
                    "version": manifest["version"],
                    "summary": run.products["summary.json"],
                    "acceptance_failed": acc["failed"],
                    "cases_failed": acc["failed_cases"],
                    "refuted": [i["name"] for i in exp["items"]
                                if i["verdict"] == "refuted"]})
    return 0 if not _acceptance_failed(acc) else 1


def _products_for_readonly(args: argparse.Namespace) -> tuple[dict[str, Any], bool, str]:
    """读 `out/`；缺产物则在内存里重跑（**绝不落盘**）。"""
    cfg, out = _resolve(args)
    from .emit import read_products
    from .visualize import REQUIRED
    missing = [n for n in REQUIRED if not (out / n).exists()]
    if not missing:
        return read_products(out, REQUIRED), False, ""
    run = pipeline.build(cfg)
    pipeline.finish(run)
    # `finish` 会放一个**占位**版本清单（渲染器需要它）。
    # 这里把版本字段改成「内存重跑」的说明，免得报告里印出一个看起来像
    # 真版本号的占位字符串。
    vj = run.products.get("version.json")
    if isinstance(vj, dict) and str(vj.get("version", "")).startswith("(占位"):
        vj["version"] = "(内存重跑，未写盘)"
        vj["note"] = "本次是内存重跑，未写盘。"
    return run.products, True, f"缺产物 {missing}，本次是内存重跑，未写盘"


def cmd_visualize(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        products, reran, note = _products_for_readonly(args)
        md = render_markdown(products)
        html = render_html(products)
        target = getattr(args, "target", None)
        if target is not None:
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "report.md").write_text(md, encoding="utf-8", newline="\n")
            (Path(target) / "report.html").write_text(html, encoding="utf-8",
                                                      newline="\n")
            written = [str(Path(target) / "report.md"),
                       str(Path(target) / "report.html")]
        else:
            written = []
        rows = []
        for name, text in (("report.md", md), ("report.html", html)):
            p = out / name
            existing = p.read_text(encoding="utf-8") if p.exists() else None
            rows.append({"file": name, "exists": existing is not None,
                         "identical": existing == text,
                         "rebuilt_bytes": len(text.encode("utf-8"))})
        mismatch = _mismatched_reports(rows)
        result = {
            "cmd": "visualize", "out": str(out), "read_only": not written,
            "in_memory_rerun": reran, "note": note or "只读：未写盘。",
            "files": rows, "all_identical": all(r["identical"] for r in rows),
            "mismatch": mismatch,
            "written": written,
        }
        if not json_mode:
            print(f"out/ 报告与重建结果："
                  f"{'逐位相同' if result['all_identical'] else '**不一致**'}")
            for r in rows:
                print(f"  {r['file']:12s} 存在={r['exists']} "
                      f"相同={r['identical']} 重建 {r['rebuilt_bytes']} 字节")
            print(result["note"])
            for w in written:
                print(f"  写出 {w}")
    if json_mode:
        _emit_json(result)
    return 0 if not result["mismatch"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        cfg, out = _resolve(args)
        products, reran, note = _products_for_readonly(args)
        acc = products["acceptance.json"]
        exp = products["expectations.json"]
        if not json_mode:
            print(f"门槛 {acc['n_passed']}/{acc['n']} 通过")
            for c in acc["checks"]:
                mark = "OK  " if c["passed"] else "FAIL"
                print(f"  [{mark}] {c['name']:34s} {c['detail']}")
            if acc.get("cases"):
                print(f"对抗用例 {acc['n_cases_passed']}/{acc['n_cases']} 成立"
                      f"（**不可调**的不变量断言）")
                for c in acc["cases"]:
                    if not c["ok"]:
                        print(f"  [CASE FAIL] {c['name']}: {c['detail']}")
            print(f"期望：{exp['n_held']} 成立 / {exp['n_refuted']} 被推翻 / "
                  f"{exp['n_unmeasurable']} 算不出来")
            for i in exp["items"]:
                if i["verdict"] != "held":
                    print(f"  [{i['verdict']:12s}] {i['name']}: {i['detail']}")
            print(note or "只读：未写盘。")
    if json_mode:
        _emit_json({"cmd": "check", "out": str(out), "in_memory_rerun": reran,
                    "note": note or "只读：未写盘。",
                    "acceptance": acc, "expectations": exp})
    return 0 if not acc["failed"] and not acc.get("n_cases_failed") else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """确定性 + 产物充分性。**两次 run 只许在时间戳与耗时上不同。**"""
    json_mode = bool(getattr(args, "json", False))
    cfg, _out = _resolve(args)
    result: dict[str, Any] = {}
    with _quiet_prose(json_mode):
        tmp = Path(tempfile.mkdtemp(prefix="kernelfactory-verify-"))
        try:
            versions = []
            dirs = []
            for _i in (1, 2):
                d = tmp / f"run{_i}"
                run = pipeline.build(cfg)
                pipeline.finish(run)
                manifest = emit(
                    d, run.products, pipeline.VERSIONED_FILES,
                    pipeline.UNVERSIONED_FILES,
                    extra=lambda p: {"report.md": render_markdown(p),
                                     "report.html": render_html(p)},
                )
                versions.append(manifest["version"])
                dirs.append(d)

            all_files = sorted({p.name for d in dirs for p in d.iterdir()})
            rows = []
            for name in all_files:
                a = (dirs[0] / name).read_bytes()
                b = (dirs[1] / name).read_bytes()
                rows.append({"file": name, "identical": a == b, "bytes": len(a)})

            # 产物充分性：从产物重建报告 == 直接渲染
            from .emit import read_products
            from .visualize import REQUIRED
            p1 = read_products(dirs[0], REQUIRED)
            suff = {
                "report.md": render_markdown(p1) == (dirs[0] / "report.md")
                .read_text(encoding="utf-8"),
                "report.html": render_html(p1) == (dirs[0] / "report.html")
                .read_text(encoding="utf-8"),
            }

            expect_diff = {"version.json", "benchmark.json",
                           "report.md", "report.html"}
            unexpected = [r["file"] for r in rows
                          if not r["identical"] and r["file"] not in expect_diff]
            result = {
                "cmd": "verify",
                "version_1": versions[0], "version_2": versions[1],
                "same_version": versions[0] == versions[1],
                "files": rows,
                "expected_to_differ": sorted(expect_diff),
                "expected_to_differ_why": {
                    "version.json": "created_at 时间戳（唯一允许印时间戳的地方）",
                    "benchmark.json": "实测耗时",
                    "report.md": "视图，且正文里嵌了耗时表",
                    "report.html": "视图，且正文里嵌了耗时表",
                },
                "unexpectedly_different": unexpected,
                "products_sufficient": suff,
                "products_sufficient_all": all(suff.values()),
                "ok": (versions[0] == versions[1] and not unexpected
                       and all(suff.values())),
            }
            if not json_mode:
                print(f"版本 {versions[0]} / {versions[1]} "
                      f"（相同={result['same_version']}）")
                for r in rows:
                    tag = "逐位相同" if r["identical"] else "**不同**"
                    print(f"  {r['file']:18s} {tag:8s} {r['bytes']} 字节")
                print(f"预期不同的文件：{sorted(expect_diff)}")
                print(f"**意外不同**的文件：{unexpected or '（无）'}")
                print(f"产物充分性（从产物重建报告 == 直接渲染）：{suff}")
                print("结论：" + ("**全部通过**" if result["ok"] else "**有核对未通过**"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if json_mode:
        _emit_json(result)
    return 0 if result.get("ok") else 1


def cmd_show(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    with _quiet_prose(json_mode):
        products, reran, note = _products_for_readonly(args)
        s = products["summary.json"]
        if not json_mode:
            print(f"版本 {products['version.json']['version']}")
            print(f"算子 {s['n_ops']} 个，空间总计 {s['space_size_total']} 个 kernel")
            for op in sorted(s["space_size_per_op"]):
                b = s["baseline"][op]
                print(f"  {op:10s} 基线 {b['label']}：{b['latency_us']} µs"
                      f"（{b['bottleneck']}）")
            print(f"头条漂移比（假装 fp32 ÷ 诚实 bf16）："
                  f"{s['headline_drift_ratio']}")
            print(f"门 1 放行、门 2 拒掉：{s['n_gate1_only_anywhere']} 个非法 kernel")
            print(f"安全窗口非空：{s['safe_window_nonempty']}")
            print(f"阶梯买到速度：{s['ladder_best_speedup_oracle']}")
            print(f"期望 {s['expectations']['n_held']} 成立 / "
                  f"{s['expectations']['n_refuted']} 被推翻")
            print(f"门槛 {s['acceptance']['n_passed']}/{s['acceptance']['n']}")
            print(f"对抗用例 "
                  f"{s['acceptance']['n_cases'] - (s['acceptance']['n_cases_failed'] or 0)}"
                  f"/{s['acceptance']['n_cases']}")
            print(note or "只读：未写盘。")
    if json_mode:
        _emit_json({"cmd": "show", "in_memory_rerun": reran,
                    "note": note or "只读：未写盘。",
                    "summary": products["summary.json"]})
    return 0


def cmd_dump_config(args: argparse.Namespace) -> int:
    """把当前生效的配置写回磁盘。

    ⚠️ 改了 `config.py` 里的 `DEFAULTS` 之后**必须**跑这条：
    `configs/default.json` 是它的落盘副本，而 `deep_merge` 里**列表整体替换、
    磁盘优先** ⇒ 磁盘配置会让代码里的新声明**静默失效**
    （「改了声明、重跑，数字一字未变」，附录 C 204）。
    `load_config` 里的 `no_stale_config_keys` 只挡「磁盘上多出来的键」，
    挡不住「磁盘上的旧列表盖掉了新列表」。
    """
    json_mode = bool(getattr(args, "json", False))
    cfg, _out = _resolve(args)
    path = Path(getattr(args, "config", None) or (args.root / "configs" / "default.json"))
    with _quiet_prose(json_mode):
        dump_config(cfg, path)
        if not json_mode:
            print(f"已写出 {path}")
    if json_mode:
        _emit_json({"cmd": "dump-config", "path": str(path)})
    return 0


_COMMANDS = {"run": cmd_run, "visualize": cmd_visualize, "check": cmd_check,
             "verify": cmd_verify, "show": cmd_show,
             "dump-config": cmd_dump_config}


def main(argv: list[str] | None = None) -> int:
    # 必须在 `parse_args` **之前**：argparse 打印 `--help` 时就会写 stdout，
    # 等命令跑起来再改编码已经晚了（第一次 `--help` 就会崩）。
    _configure_output()
    parser = _build_parser()
    try:
        args = _anchor_args(parser.parse_args(argv))
    except SystemExit as exc:                        # argparse 自己会退出
        return int(exc.code or 0)
    try:
        return _COMMANDS[args.cmd](args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2


__all__ = ["main", "ROOT", "_anchor_args", "_build_parser"]
