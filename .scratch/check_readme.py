"""README 数字核对：附录 B 里的每一个数，都必须能**从最终 `out/` 重新导出**。

为什么要写这个
--------------
「每个数字都从最终 `out/` 重新导出」是一条**自觉**，而自觉会失效：
改一次配置、重跑一次流水线，附录 B 就有一半的数字变成**上一次运行的读数**，
而它们看起来完全正常。本项目已经栽过一次（C.300：手抄的附录表汇总数对得上、
行对不上）。

做法
----
1. 把 `out/*.json` 与 `.scratch/mutation.json` 里的**全部数字**收进一个池子；
2. 从 README 的**附录 B** 里抠出全部数字 token；
3. 每个 token 必须能被池子里某个数**按它自己的精度舍入**得到；
   否则必须落在一张**显式声明**的表里（手测 / 派生），且表里每一项都要写理由；
4. 再核几条**复合断言**（版本号、指纹、门槛计数…）确实出现在 README 里。

判据是「**能被舍入得到**」而不是「字符串相等」—— 因为附录 B 会做
`%.4f` / `%.3e` 这类排版舍入，而产物里是全精度。舍入区间用**有效数字位数**
推出来：一个写了 `s` 位有效数字的 token，容许区间是它自己的半个末位。

用法：`python .scratch/check_readme.py`
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

from _output import configure_utf8  # noqa: E402  （脚本目录在 sys.path 上）

#: 数字 token：前后不许紧贴字母或小数点（于是 `C.317`、`v1-1e5ab0…`、`L6` 都不算）
TOKEN = re.compile(r"(?<![\w.])[-−–]?\d+(?:\.\d+)?(?:[eE][-−]?\d+)?(?![\w.])")

#: 手测 / 派生 —— **每一项都必须写理由**，否则这张表会变成「随手加数字的地方」
MANUAL: dict[str, str] = {
    "12.1": "`run` 的墙钟耗时（subprocess 计时），产物里只有流水线内部计时",
    "23.0": "`verify` 的墙钟耗时（它跑两次 run），不在产物里",
    "0.41": "`visualize` 的墙钟耗时（只读重建），不在产物里",
    "17": "`ladder.json` 从 3.4 MB 降到 2.81 MB 的降幅百分比（3.4 来自历史运行）",
    "15": "`out/` 里的文件个数（跑只读命令前后比对用），不写进产物",
    "1e-18": "举例用的量级（说明「恰好为 0」为什么不能用近似判），不是读数",
    "0.123456": "变异测试里**注入的**字面量（它本来就不是产物里的数）",
    "0.61": "历史读数：探针形状改之前那一版配置下的超体积缺口（见附录 C.315）",
}

#: 交叉引用豁免：`C.NNN` 这种写法是**章节引用**，不是测量值。
#: 附录 B 里引用附录 C 的编号时一律写成 `C.NNN`（点号形式），
#: 于是 TOKEN 的 `(?<![\w.])` 负向后顾会把它排除掉 —— **全文档一个写法**，
#: 这样「哪些数字需要解释」这条规则才不需要为交叉引用开口子。
#: （第一版有两处写成「附录 C 323」，空格形式会被当成数字 token 抓出来。）


def walk_numbers(node, sink: list) -> None:
    """把产物里的数字收进池子 —— **包括嵌在文案里的数字**。

    `detail` / `note` / `created_at` 这些字符串也是产物的一部分，
    里面的数字（「相对增益 2.3205」、ISO 时间戳的 `2026`）同样是产物里的读数。
    只收数字叶子会漏掉它们，然后逼着人去手抄一张例外表 —— 那张表迟早会失控。
    """
    if isinstance(node, dict):
        for v in node.values():
            walk_numbers(v, sink)
    elif isinstance(node, list):
        for v in node:
            walk_numbers(v, sink)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        sink.append(float(node))
    elif isinstance(node, str):
        for tok in TOKEN.findall(node):
            try:
                sink.append(float(tok.replace("−", "-").replace("–", "-")))
            except ValueError:
                pass


def sig_interval(tok: str) -> tuple[float, float]:
    """token 自己的「半个末位」区间 —— 由它写了多少位有效数字推出来。"""
    t = tok.replace("−", "-").replace("–", "-")
    val = float(t)
    body = t.lower()
    mant = body.split("e")[0].lstrip("-")
    if "." in mant:
        int_part, frac = mant.split(".")
    else:
        int_part, frac = mant, ""
    digits = (int_part + frac).lstrip("0")
    s = len(digits) or 1
    if val == 0.0:
        return (-5e-13, 5e-13)
    exp = int(body.split("e")[1]) if "e" in body else 0
    # 数值量级：0.9848 → 9.848e-1
    import math
    e10 = math.floor(math.log10(abs(val)))
    half = 0.5 * 10 ** (e10 - s + 1)
    return (val - half, val + half)


def cli_contract(text: str) -> list[str]:
    """把 README 的 A.3 表**对着 `--help` 机器核** —— 这一条是 C.323 的机器化。

    ⚠️ 为什么不能靠「跑一遍看文件变没变」来核：`dump-config` 写回的是**同一份内容**，
    所以任何基于 md5 的核对都会把它判成只读。**哈希能证明「没变」，不能证明「没写」。**
    ⇒ 命令的读写属性只能读代码 / `--help` 来核。
    """
    import subprocess
    # 从 A.3 表里抠出「命令 → 那一行」
    rows: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("## A.3 CLI 契约"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if in_table:
            m = re.match(r"\|\s*`([a-z-]+)`\s*\|(.*?)\|", line)
            if m:
                rows[m.group(1)] = m.group(2)
    if not rows:
        return ["A.3 表没解析出来 —— 契约没被核到（这本身就是失败）"]

    write_verbs = ("写回", "落盘", "写到", "写出", "写入", "写盘")
    problems: list[str] = []
    for cmd in sorted(rows):
        cell = rows[cmd]
        proc = subprocess.run([sys.executable, "main.py", cmd, "--help"],
                              cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8")
        if proc.returncode != 0:
            problems.append(f"`{cmd} --help` rc={proc.returncode}")
            continue
        # 子命令的 `--help` = usage 块（**可能折行**）+ 空行 + 描述 + 空行 + options。
        # ⚠️ 第一版只跳过「以 usage: 开头的那一行」，于是折行出来的
        #    `[--target TARGET]` 被当成描述，里面的「落盘」触发了假阳性。
        lines = [ln.strip() for ln in proc.stdout.splitlines()]
        i = next((k for k, ln in enumerate(lines) if ln.startswith("usage:")), 0)
        while i < len(lines) and lines[i]:
            i += 1
        start = i + 1
        end = next((k for k, ln in enumerate(lines)
                    if k >= start and ln.startswith("options:")), len(lines))
        desc = " ".join(ln for ln in lines[start:end] if ln)

        # **判据用两个显式标记**，两边必须用同一套词：
        #   `会写盘` = 默认行为是写；`只读` = 默认行为是不写。
        # 这样「条件写」（`visualize --target`）不需要为它开特例 ——
        # 它的默认行为就是只读，`--target` 的写在 options 里单列。
        doc_write = "会写盘" in desc
        doc_readonly = "只读" in desc
        readme_write = "会写盘" in cell
        readme_readonly = "只读" in cell
        if doc_write == doc_readonly:
            problems.append(f"`{cmd}`：`--help` 里 `会写盘`/`只读` 两个标记"
                            f"必须**恰好出现一个** → {desc!r}")
        if readme_write == readme_readonly:
            problems.append(f"`{cmd}`：README 的「写盘」列里 `会写盘`/`只读` "
                            f"必须**恰好出现一个** → {cell.strip()!r}")
        if doc_write != readme_write:
            problems.append(
                f"`{cmd}`：README 说「{'会写盘' if readme_write else '只读'}」，"
                f"`--help` 说「{'会写盘' if doc_write else '只读'}」 → {desc!r}")
        if doc_write and not any(v in desc for v in write_verbs):
            problems.append(f"`{cmd}`：标了「会写盘」但没说写到哪 → {desc!r}")
    return problems


def main() -> int:
    configure_utf8()
    products = {f.name: json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((ROOT / "out").glob("*.json"))}
    if not products:
        print("❌ out/ 里没有产物 —— 先跑 `python main.py run`")
        return 1

    pool: list[float] = []
    for name, doc in products.items():
        pool.append(float(1))          # 占位，避免空池
        walk_numbers(doc, pool)
    mut_path = ROOT / ".scratch" / "mutation.json"
    mut = None
    if mut_path.exists():
        mut = json.loads(mut_path.read_text(encoding="utf-8"))
        walk_numbers(mut, pool)
    pool = sorted(set(pool))

    # ---- 派生量（由池子里的数**算出来**，不是手抄的）----------------------
    derived: dict[str, str] = {}

    def derive(label: str, value: float) -> None:
        derived[f"{value:.10g}"] = label

    s = products["summary.json"]
    ladder_bytes = (ROOT / "out" / "ladder.json").stat().st_size
    derive("ladder.json 的 MB 数", round(ladder_bytes / 1e6, 2))
    for op, arr in s["ladder_best_speedup_oracle"].items():
        derive(f"{op} 阶梯首末级加速比之比", arr[-1] / arr[0])
    wc = json.loads((ROOT / "out" / "benchmark.json").read_text(encoding="utf-8"))["wallclock"]
    for op, d in wc.items():
        r = d["rows"][0]
        derive(f"{op} 实测/解析 比值", r["measured_eval_us"] / r["analytic_latency_us"])
        # 正文会把 matmul 的实测耗时从 µs 换成 ms 讲（「一次要 33.7 ms」），
        # 单位换算是**派生量**，不是新读数 —— 不补它就得为它开例外。
        derive(f"{op} 实测耗时的毫秒数", r["measured_eval_us"] / 1000.0)
    if mut:
        kc = mut["kill_count"]
        kinds: dict[str, list[int]] = {}
        for label, n in kc.items():
            kinds.setdefault(label.split(":")[0], []).append(n)
        for k, ns in kinds.items():
            derive(f"变异类别 {k} 的个数", len(ns))
            derive(f"变异类别 {k} 有杀伤的个数", sum(1 for n in ns if n))
            derive(f"变异类别 {k} 弄红条数合计", sum(ns))
        derive("通用电池能红的用例数",
               mut["n_cases"] - len(mut["never_reddened_by_battery"]))
        derive("实际跑了的变异数", mut["n_mutations"] - len(mut["vacuous_mutations"]))
        derive("有杀伤的变异数", sum(1 for n in kc.values() if n))
        derive("无杀伤的变异数",
               mut["n_mutations"] - len(mut["vacuous_mutations"])
               - sum(1 for n in kc.values() if n))
    # 分列的前沿点数 / 不同权衡点数
    for op, d in products["space.json"]["per_op"].items():
        derive(f"{op} 前沿点数", d["n_frontier_at_tau0"])
        derive(f"{op} 不同权衡点数", d["n_frontier_distinct_at_tau0"])
    # 空间总量（笛卡尔积 / 剪掉 / 合计）与默认容差下 greedy 的超体积缺口
    tot_raw = tot_pruned = tot_net = 0
    for op, d in products["space.json"]["per_op"].items():
        tot_raw += d["n_raw"]
        tot_pruned += d["n_pruned"]
        tot_net += d["n_genomes"]
    derive("三个算子的笛卡尔积合计", tot_raw)
    derive("三个算子剪掉的合计", tot_pruned)
    derive("三个算子剪后合计", tot_net)
    lc = products["ladder.json"]["per_op"]
    for op, cell in lc.items():
        rung = cell["reject@1e-05"]
        hv = {n: c["hypervolume"] for n, c in rung["cells"].items()}
        derive(f"{op} 默认容差下 greedy 的超体积缺口",
               1.0 - hv["greedy"] / hv["oracle"])
        derive(f"{op} 默认容差下 greedy 的延迟缺口",
               1.0 - rung["cells"]["greedy"]["best"]["speedup"]
               / rung["cells"]["oracle"]["best"]["speedup"])

    derived_vals = [float(k) for k in derived]

    # ---- 抠 README 附录 B 的数字 ----------------------------------------
    text = README.read_text(encoding="utf-8")
    start = text.index("# 附录 B｜实测结果")
    end = text.index("# 附录 C｜踩过的坑")
    section = text[start:end]
    tokens = sorted(set(TOKEN.findall(section)))

    def matched(tok: str) -> bool:
        lo, hi = sig_interval(tok)
        for v in pool:
            if lo <= v <= hi:
                return True
        for v in derived_vals:
            if lo <= v <= hi:
                return True
        return False

    unmatched = [t for t in tokens if not matched(t)]
    manual_hit = [t for t in unmatched if t in MANUAL]
    bad = [t for t in unmatched if t not in MANUAL]

    # ---- 复合断言：这几条必须出现，且必须与产物一致 ----------------------
    v = products["version.json"]["version"]
    claims = [
        ("版本号", v),
        ("协议指纹", products["config.json"]["protocol_digest"]),
        ("门槛计数", f"{s['acceptance']['n_passed']} / {s['acceptance']['n']}"),
        ("对抗用例计数", f"{s['acceptance']['n_cases']} / {s['acceptance']['n_cases']}"),
        ("期望计数",
         f"{s['expectations']['n_held']} 成立 / {s['expectations']['n_refuted']} 被推翻"),
        ("空间总计", str(s["space_size_total"])),
    ]
    missing_claims = [(lab, val) for lab, val in claims if val not in text]

    print(f"产物 {len(products)} 个；数字池 {len(pool)} 个数；派生 {len(derived)} 个数")
    print(f"附录 B 里的数字 token：{len(tokens)} 个")
    print(f"  能从产物/派生量解释：{len(tokens) - len(unmatched)}")
    print(f"  落在手测表里：{len(manual_hit)}  → {manual_hit}")
    print(f"  **解释不了**：{len(bad)}  → {bad}")
    print()
    if missing_claims:
        print("❌ 复合断言缺失：")
        for lab, val in missing_claims:
            print(f"   - {lab}：README 里没有 {val!r}")
    else:
        print("✅ 复合断言全部命中：")
        for lab, val in claims:
            print(f"   - {lab}: {val}")

    # 过期版本号扫描：README 里出现的任何 v1-<hex12> 都必须是当前版本
    stale = sorted({m for m in re.findall(r"v1-[0-9a-f]{12}", text) if m != v})
    if stale:
        print(f"\n❌ README 里有**过期版本号**：{stale}（当前 {v}）")
    else:
        print(f"\n✅ 没有过期版本号（README 里只出现 {v}）")

    cli_bad = cli_contract(text)
    if cli_bad:
        print("\n❌ A.3 的「写盘」列与 `--help` 不一致（C.323 的机器化）：")
        for x in cli_bad:
            print(f"   - {x}")
    else:
        print("✅ A.3 的「写盘」列与每个子命令的 `--help` 逐条一致")

    ok = not bad and not missing_claims and not stale and not cli_bad
    print("\n结论：" + ("**全部通过**" if ok else "**有核对未通过**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
