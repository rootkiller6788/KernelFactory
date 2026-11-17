"""变异测试：证明 `invariants.py` 里的每条对抗用例**都能红**。

为什么必须有这一步
------------------
「58/58 成立」这个数字本身没有信息量 —— 一条恒真的用例也是 58/58 里的一个。
真正要回答的是：**有没有哪条用例无论产物怎么坏都不会红？** 那才是恒真判据。

做法（两段）
------------
① **通用变异电池**：对 `out/` 的产物树做机械破坏 —— 删文件、删键、改数、
   翻布尔、截断/重复/反转列表、换兄弟值。每做一次破坏就跑完整 58 条用例，
   记下哪几条红了。不针对任何具体用例，所以不会「照着用例写答案」。
② **定点补测**：通用电池没能弄红的用例，逐条手写一个最小破坏。

判据
----
- 每条用例至少被一种破坏弄红 → 它**有区分能力**（不是恒真）。
- 每种破坏至少弄红一条用例 → 它**有杀伤力**（不是无效变异）。
- 一次都弄不红的用例 → 单独列出来，**这是要报的问题，不是要藏的问题**。

用法：`python .scratch/prove_cases_can_fail.py [--json out.json]`
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kernelfactory.invariants import _CASES, run_cases          # noqa: E402

SEED = 20261002
BUDGET_S = 900.0


# ----------------------------------------------------------------- 产物装载
def load_products(out_dir: Path) -> dict[str, Any]:
    return {f.name: json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(out_dir.glob("*.json"))}


# ----------------------------------------------------------------- 路径枚举
def walk(node: Any, path: tuple, acc: dict[str, list]) -> None:
    if isinstance(node, dict):
        acc["dict"].append(path)
        for k in node:
            walk(node[k], path + (k,), acc)
    elif isinstance(node, list):
        acc["list"].append(path)
        for i, v in enumerate(node):
            walk(v, path + (i,), acc)
    elif isinstance(node, bool):
        acc["bool"].append(path)
    elif isinstance(node, (int, float)):
        acc["num"].append(path)
    elif isinstance(node, str):
        acc["str"].append(path)


def sample(pool: list, n: int, rng: random.Random) -> list:
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


# ----------------------------------------------------------------- 变异
def get(products: dict, path: tuple) -> Any:
    node: Any = products
    for k in path:
        node = node[k]
    return node


def set_at(products: dict, path: tuple, value: Any) -> None:
    node: Any = products
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


def del_at(products: dict, path: tuple) -> None:
    node: Any = products
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]


def _repr(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class Mutation:
    """一次破坏：`apply` 改、`undo` 还原（原地改，不深拷 3MB 产物）。

    三种 op，语义各自闭合 —— `path` 一律是**被删/被改的那个节点本身**：
      - `set`      ：把 `path` 处的值换成 `new`
      - `del_key`  ：把 `path` 处的键整个删掉
      - `del_file` ：把 `path[0]` 这个产物整个删掉
    """

    def __init__(self, label: str, path: tuple, undo: Any, kind: str) -> None:
        self.label = label
        self.path = path
        self._undo = undo
        self.kind = kind

    def before(self, products: dict) -> Any:
        if self._undo["op"] == "del_file":
            return products.get(self.path[0])
        return get(products, self.path)

    def apply(self, products: dict) -> None:
        op = self._undo["op"]
        if op == "set":
            set_at(products, self.path, self._undo["new"])
        elif op == "del_key":
            del_at(products, self.path)
        elif op == "del_file":
            del products[self.path[0]]

    def undo(self, products: dict) -> None:
        op = self._undo["op"]
        if op == "set":
            set_at(products, self.path, self._undo["old"])
        elif op == "del_key":
            set_at(products, self.path, self._undo["old"])
        elif op == "del_file":
            products[self.path[0]] = self._undo["old"]


def build_mutations(products: dict, rng: random.Random,
                    per_kind: int) -> list[Mutation]:
    acc: dict[str, list] = {k: [] for k in
                            ("dict", "list", "bool", "num", "str")}
    walk(products, (), acc)
    muts: list[Mutation] = []

    # ① 整个产物文件消失（缺文件）
    for name in sorted(products):
        muts.append(Mutation(f"drop_file:{name}", (name,),
                             {"op": "del_file", "old": products[name]},
                             "drop_file"))

    # ② 键消失（父 dict 至少有 2 个键，否则等于删整个节点）
    pools = [p for p in acc["dict"] if len(get(products, p)) >= 2]
    for p in sample(pools, per_kind, rng):
        d = get(products, p)
        key = rng.choice(sorted(d))
        muts.append(Mutation(
            f"drop_key:{'.'.join(map(str, p + (key,)))}", p + (key,),
            {"op": "del_key", "old": d[key]}, "drop_key"))

    # ③ 数字归零 / 取负 / 微扰（float 才微扰，int 微扰会变成 float 反而更假）
    nums = [p for p in acc["num"] if abs(get(products, p)) < 10 ** 6]
    for p in sample(nums, per_kind, rng):
        v = get(products, p)
        muts.append(Mutation(f"zero:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": v, "new": 0}, "zero"))
        muts.append(Mutation(f"negate:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": v, "new": -v}, "negate"))
        if isinstance(v, float) and v != 0.0:
            muts.append(Mutation(f"nudge:{'.'.join(map(str, p))}", p,
                                 {"op": "set", "old": v, "new": v * 1.0001},
                                 "nudge"))

    # ④ 布尔翻转
    for p in sample(acc["bool"], per_kind, rng):
        v = get(products, p)
        muts.append(Mutation(f"flip:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": v, "new": not v}, "flip"))

    # ⑤ 字符串清空
    strs = [p for p in acc["str"] if len(get(products, p)) > 3]
    for p in sample(strs, per_kind, rng):
        v = get(products, p)
        muts.append(Mutation(f"blank:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": v, "new": ""}, "blank"))

    # ⑥ 列表：截断 / 重复末项 / 反转
    for p in sample([q for q in acc["list"] if len(get(products, q)) >= 2],
                    per_kind, rng):
        lst = get(products, p)
        muts.append(Mutation(f"truncate:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": lst,
                              "new": lst[:-1]}, "truncate"))
        muts.append(Mutation(f"dup_last:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": lst,
                              "new": lst + [copy.deepcopy(lst[-1])]},
                             "dup_last"))
        muts.append(Mutation(f"reverse:{'.'.join(map(str, p))}", p,
                             {"op": "set", "old": lst,
                              "new": list(reversed(lst))}, "reverse"))

    # ⑦ 兄弟值互换（同一 dict 下两个数字）
    sib = [p for p in acc["dict"]
           if sum(1 for v in get(products, p).values()
                  if isinstance(v, (int, float))
                  and not isinstance(v, bool)) >= 2]
    for p in sample(sib, per_kind, rng):
        d = get(products, p)
        ks = sorted(k for k, v in d.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool))
        a, b = ks[0], ks[1]
        old = (d[a], d[b])
        muts.append(Mutation(f"swap:{'.'.join(map(str, p))}.{a}<->{b}", p,
                             {"op": "set", "old": d,
                              "new": {**d, a: old[1], b: old[0]}}, "swap"))
    return muts


# ----------------------------------------------------------------- 定点补测
class Targeted:
    """通用电池弄不红的用例，逐条手写最小破坏。

    `expect_red`   ：必须被弄红的用例（**少一条就算漏网**）
    `expect_green` ：必须**不**被弄红的用例（反例对照 —— 证明这条用例
                     不是「见到某个词就报」，它有边界）
    """

    def __init__(self, label: str, expect_red: tuple, apply, undo,
                 expect_green: tuple = (), note: str = "") -> None:
        self.label = label
        self.expect_red = expect_red
        self.expect_green = expect_green
        self._apply = apply
        self._undo = undo
        self.note = note

    def apply(self, products: dict) -> None:
        self._apply(products)

    def undo(self, products: dict) -> None:
        self._undo(products)


def build_targeted() -> list[Targeted]:
    out: list[Targeted] = []

    # ① 往进摘要的产物里塞一个**耗时读数**（float）
    def add_float(p):
        p["summary.json"]["timings"] = {"evaluate": 0.123456}

    out.append(Targeted(
        "inject:timings_float",
        ("no_elapsed_time_key_in_the_versioned_products",),
        add_float, lambda p: p["summary.json"].pop("timings", None),
        note="把 `timings` 顺手挂到 summary 上 —— 正是这条用例要防的事"))

    # ② 同一个位置塞一个**整数**耗时（int 也是读数，不是基数）
    def add_int(p):
        p["summary.json"]["elapsed_s"] = 8

    out.append(Targeted(
        "inject:elapsed_int",
        ("no_elapsed_time_key_in_the_versioned_products",),
        add_int, lambda p: p["summary.json"].pop("elapsed_s", None),
        note="整数耗时也是读数 —— 判据不能只认 float"))

    # ③ **反例对照**：塞一个「名字里有 timing 但它是基数」的键 ——
    #    这条用例**必须放行**，否则它就是在按词报错（实测第一次上线
    #    就是这么误报了 `expectations.json.n_timing` 三处）。
    def add_count(p):
        p["summary.json"]["n_timing_steps"] = 3

    out.append(Targeted(
        "inject:timing_count(反例对照)",
        (), add_count, lambda p: p["summary.json"].pop("n_timing_steps", None),
        expect_green=("no_elapsed_time_key_in_the_versioned_products",),
        note="**基数不是读数** —— 这条必须绿，否则判据退化成按词匹配"))

    # ④ 往进摘要的产物里塞一个 ISO 时间戳
    def add_ts(p):
        p["summary.json"]["created_at"] = "2026-10-02T21:53"

    out.append(Targeted(
        "inject:timestamp",
        ("no_timestamp_leaks_into_versioned_products",),
        add_ts, lambda p: p["summary.json"].pop("created_at", None),
        note="时间戳的唯一出口是 `version.json`"))

    # ⑤ 往产物里塞一个**不可 JSON 序列化**的对象
    def add_bad(p):
        p["summary.json"]["bad"] = {1, 2, 3}

    out.append(Targeted(
        "inject:nonserializable",
        ("products_are_json_serializable",),
        add_bad, lambda p: p["summary.json"].pop("bad", None),
        note="一个 set 混进产物 ⇒ `emit` 会炸在写盘那一刻"))

    # ⑥ **改源码**：往 matmul 内核里加一句 `gen.memory_layout` 的读取。
    #    内核从此「读了」一个没进缓存键的基因 ⇒「键相同」不再蕴含「输出相同」。
    #    这条用例扫的是 `ops.py` 的**磁盘文本**，所以它只可能被**源码变异**
    #    弄红 —— 产物怎么改都碰不到它。
    #    ⚠️ 第一版我改的是 `CACHE_KEY_COMPONENTS["matmul"]`，**没红** ——
    #    因为用例的 `covered` 来自**已导入模块的常量**，改磁盘文件不影响它；
    #    只有 AST 扫出来的 `read` 集才来自磁盘。这本身就是一条坑：
    #    「改文件」与「改进程里的模块」是两件事，在同一个进程里必须分清。
    ops = ROOT / "src" / "kernelfactory" / "ops.py"
    original = ops.read_text(encoding="utf-8")
    needle = ("    if quantize:\n"
              "        a = [round_to_bits(v, bits) for v in ins.x]")
    assert needle in original, f"源码变异锚点没找到：{needle!r}"
    planted = ("    _ = gen.memory_layout      # 变异：读了一个没进缓存键的基因\n"
               + needle)

    def patch_src(p):
        ops.write_text(original.replace(needle, planted), encoding="utf-8")

    out.append(Targeted(
        "source:kernel_reads_uncached_gene",
        ("cache_key_covers_every_gene_the_kernel_reads",),
        patch_src, lambda p: ops.write_text(original, encoding="utf-8"),
        note="扫源码的用例只能被源码变异弄红 —— 产物改不动它"))
    return out


def run_targeted(base: dict, out_dir: Path) -> tuple[int, list[str]]:
    print("\n" + "-" * 72)
    print("定点补测（通用电池弄不红的那几条）")
    print("-" * 72)
    leaks: list[str] = []
    for t in build_targeted():
        snap = copy.deepcopy(base)
        t.apply(snap)
        try:
            res = run_cases(snap)
            red = {c["name"] for c in res["cases"] if not c["ok"]}
        finally:
            t.undo(snap)
        hit = sorted(set(t.expect_red) & red)
        miss = sorted(set(t.expect_red) - red)
        broke = sorted(set(t.expect_green) & red)
        tag = "OK  " if not miss and not broke else "FAIL"
        print(f"  [{tag}] {t.label:<36} 弄红 {hit or '（无）'}")
        if t.expect_green:
            print(f"         反例对照（必须绿）：{sorted(t.expect_green)} "
                  f"→ {'绿 ✅' if not broke else '红 ❌'}")
        if miss:
            leaks.append(f"{t.label} 没弄红 {miss}")
        if broke:
            leaks.append(f"{t.label} 误伤了 {broke}（判据太宽）")
    return (0 if not leaks else 1), leaks


# ----------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--per-kind", type=int, default=40)
    ap.add_argument("--targeted-only", action="store_true",
                    help="只跑定点补测（改代码后快速回归用）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    base = load_products(out_dir)
    rng = random.Random(SEED)

    print(f"产物 {len(base)} 个；用例 {len(_CASES)} 条；种子 {SEED}")
    t0 = time.perf_counter()
    r_base = run_cases(copy.deepcopy(base))
    print(f"基线：{r_base['n_cases_passed']}/{r_base['n_cases']} 成立"
          f"（{(time.perf_counter() - t0) * 1000:.0f} ms）")
    if r_base["n_cases_failed"]:
        print(f"⚠️ 基线就不成立：{r_base['failed_cases']} —— 先修它，再谈变异")
        return 1

    muts = [] if args.targeted_only else build_mutations(base, rng, args.per_kind)
    print(f"变异 {len(muts)} 个（{len({m.kind for m in muts})} 类）\n")

    reddened: dict[str, str] = {}          # 用例名 → 第一个弄红它的变异
    kill_count: dict[str, int] = {}        # 变异标签 → 弄红了几条
    vacuous: list[str] = []                # 改了等于没改
    kind_kill: dict[str, set] = {}         # 变异类别 → 弄红过的用例

    for i, m in enumerate(muts, 1):
        before = _repr(m.before(base))
        m.apply(base)
        try:
            after = _repr(m.before(base))
        except (KeyError, IndexError, TypeError):
            after = "<gone>"
        if before == after:
            vacuous.append(m.label)
            m.undo(base)
            continue
        try:
            res = run_cases(base)
            red = [c["name"] for c in res["cases"] if not c["ok"]]
        finally:
            m.undo(base)
        kill_count[m.label] = len(red)
        kind_kill.setdefault(m.kind, set()).update(red)
        for name in red:
            reddened.setdefault(name, m.label)
        if i % 40 == 0:
            print(f"  ... {i}/{len(muts)}  已弄红 {len(reddened)}/{len(_CASES)}"
                  f"  用时 {time.perf_counter() - t0:.0f}s")
        if time.perf_counter() - t0 > BUDGET_S:
            print(f"  ⏱ 预算 {BUDGET_S:.0f}s 用完，停在 {i}/{len(muts)}")
            break

    never = sorted(set(_CASES) - set(reddened))
    no_kill = sorted({m.kind for m in muts} - set(kind_kill))
    dead = sorted(m.label for m in muts if kill_count.get(m.label, 0) == 0)

    print("\n" + "=" * 72)
    print("第一段｜通用变异电池（不针对任何用例）")
    print("-" * 72)
    print(f"  变异 {len(muts)} 个（{len({m.kind for m in muts})} 类）")
    print(f"  能红的用例        ：{len(reddened)}/{len(_CASES)}")
    print(f"  有杀伤力的变异类别：{len(kind_kill)}/{len({m.kind for m in muts})}"
          f"  没杀伤的：{no_kill or '（无）'}")
    print(f"  改了等于没改的变异：{len(vacuous)} 个（先剔掉，否则是白跑）")
    print(f"  跑了但一条都没弄红的变异：{len(dead)} 个 —— 随机改一个叶子大多仍自洽，"
          f"这 是 预 期 的")
    if never:
        print(f"  一次都没弄红的用例（{len(never)} 条）：")
        for n in never:
            print(f"     - {n}")

    rc, leaks = run_targeted(base, out_dir)
    for t in build_targeted():
        for name in t.expect_red:
            reddened.setdefault(name, f"[定点] {t.label}")

    still = sorted(set(_CASES) - set(reddened))
    print("\n" + "=" * 72)
    print("第二段｜合计")
    print("-" * 72)
    if args.targeted_only:
        print("（只跑了定点补测，通用电池被跳过 —— 这里的「合计」只覆盖定点那几条）")
    print(f"能红的用例：{len(reddened)}/{len(_CASES)}")
    if still and not args.targeted_only:
        print(f"❌ 仍然没被弄红：{still}")
    elif not still:
        print("✅ 每条用例都至少被一种破坏弄红 —— 没有恒真判据。")
    if leaks:
        print("❌ 定点补测的问题：")
        for x in leaks:
            print(f"   - {x}")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "seed": SEED, "n_mutations": len(muts),
            "n_cases": len(_CASES), "n_reddened": len(reddened),
            "never_reddened_by_battery": never,
            "never_reddened_at_all": still,
            "targeted_leaks": leaks,
            "kinds_without_kill": no_kill,
            "vacuous_mutations": vacuous,
            "kill_count": kill_count,
            "first_kill": reddened,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"明细已写 {args.json}")
    bad = bool(leaks) or (bool(still) and not args.targeted_only)
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
