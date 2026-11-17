"""对抗用例：**关键不变量**的运行时核对。

## 与 `acceptance.py` 的分工

- `acceptance.py` 的**门槛**是**可调**的：阈值写在 `config.acceptance` 里，
  可以被调松（那正是「门槛」的意义 —— 它是达标线）。
- 这里的**对抗用例**是**不可调**的：要么成立要么不成立，配置里没有它的阈值。
  它们不是达标线，是「**证明这个机制真的在工作**」。
  ⇒ 一条对抗用例红了，意味着**代码或产物自相矛盾**，不是「这次没达标」。

## ⚠️ 一条口径纪律（本项目最容易踩的）

`adversarial.json` 里那些 **非法变体**（`drop_tail` / `skip_normalization` /
`mantissa_truncate(b)` …）是**被测量的实验对象** —— 它们是**数据**，
不是「对抗用例」。**同一个词指了两个不同的东西，读者没法从报告内部发现。**
⇒ 本文件用「对抗用例」只指不变量断言；那些一律叫**非法变体**。

## 三条写用例的纪律

1. **用例红了，第一反应是「用例是不是写错了」。** 一个自己写错的用例比没有用例更糟 ——
   它会让人去修一个不存在的问题。
2. **用例自己抛异常也必须算失败**，不许吞（吞掉就成了一条静默失效的通道）。
3. **用例要能红。** 每条用例的 `claim` 都要写清「它在什么情况下会红」，
   答不上来的那条删掉 —— 一条恒真的用例与没有用例在输出里长得一模一样。

## 一条本项目的特殊纪律：**不许有恒真的用例**

好几条用例的判据天然容易退化成恒真（例如「同键 ⇒ 同输出」在**所有键都不同**时
必然成立）。所以它们必须**同时报一个「非空洞」读数**并在用例内部断言它：
`cache_key_equality_implies_output_equality` 会报 `n_same_key_pairs` 并要求 ≥ 1。
**少了这个读数，那条用例在最坏情况下会变成一句永远正确的话。**
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Callable

from .evaluate import KernelEvaluator, accum_steps
from .genome import GENE_ORDER, KernelGenome
from .numerics import SIGNIFICANT_BITS, expected_bits_bias
from .objective import OBJECTIVES
from .ops import (
    CACHE_KEY_COMPONENTS, ILLEGAL_VARIANTS, VARIANT_KEY_COMPONENTS, cache_key,
    drift_of, kernel_output,
)
from .pareto import dominates, hypervolume_2d
from .pipeline import VERSIONED_FILES

#: 用例名 → 求值函数。返回 `(ok, detail, observed)`。
_CASES: dict[str, Callable[[dict[str, Any]], tuple[bool, str, Any]]] = {}


def _case(name: str):
    def deco(fn):
        _CASES[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------- 小工具
def _cfg(p: dict[str, Any]) -> dict[str, Any]:
    return p["config.json"]["resolved"]


def _ops(p: dict[str, Any]) -> list[str]:
    return sorted(_cfg(p)["ops"])


def _space(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["space.json"]["per_op"][op]


def _genes(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["genes.json"]["per_op"][op]


def _adv(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["adversarial.json"]["per_op"][op]


def _ladder(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["ladder.json"]["per_op"][op]


def _tolerances(p: dict[str, Any]) -> list[float]:
    return [float(x) for x in _cfg(p)["ladder"]["tolerances"]]


def _cell(p: dict[str, Any], op: str, strategy: str, tol: float) -> dict[str, Any]:
    return _ladder(p, op)[f"{strategy}@{tol:g}"]["cells"]


def _at(accepted_at: dict[str, bool], tol: float) -> bool:
    """容差 → `accepted_at` 的键，判「这一档接受了吗」。

    ⚠️ 键是 `str(t)` 生成的（见 `adversarial.py`），**不是** `f"{t:g}"`：
    对 `0.0` 前者给 `'0.0'`、后者给 `'0'` —— 用 `:g` 查 τ=0 那一级会**永远查不到**。
    于是「τ=0 时非法变体全部被拒」这句话里，τ=0 那一格**根本没被检查**。
    **一个永远查不到的键，与「查到了且通过」在输出里长得一模一样。**
    """
    return bool(accepted_at.get(str(float(tol))))


def _illegal(p: dict[str, Any], op: str) -> list[dict[str, Any]]:
    """**全部**非法变体（尺子 + 具名 bug）—— 一个来源，不许只取一半。"""
    a = _adv(p, op)
    return list(a["variants"])


def _trial_like(d: dict[str, Any]) -> Any:
    from .evaluate import Trial
    return Trial.from_dict(d)


def _new_evaluator(p: dict[str, Any], op: str) -> KernelEvaluator:
    """从产物重建一个评估器。**只用来做数值侧的重算**（便宜：不算全空间）。"""
    return KernelEvaluator(_cfg(p), op)


# ===================================================================== 空间
@_case("space_size_equals_cartesian_product_minus_pruned")
def _i_space_size(p):
    """合法空间 = 原始笛卡尔积 − 被过滤掉的数量。

    红的情况：过滤规则把基因组**既不收下也不计数**（例如提前 `continue` 到
    另一个分支），于是「空间比预期小」这件事在产物里没有痕迹。
    """
    bad = []
    for op in _ops(p):
        s = _space(p, op)
        if s["n_raw"] - s["n_pruned"] != s["n_valid_before_dedup"]:
            bad.append(op)
        if s["n_valid_before_dedup"] - s["n_collapsed_duplicates"] != s["n_genomes"]:
            bad.append(op)
    return (not bad,
            f"{len(_ops(p))} 个算子里 {len(bad)} 个的加减法对不上：{bad}",
            {"mismatched_ops": bad})


@_case("prune_reasons_account_for_every_dropped_genome")
def _i_prune_reasons(p):
    """过滤原因分布之和 == 被过滤掉的总数。

    红的情况：新增一条过滤规则但忘了记原因 —— 于是原因表看起来完整，
    而它少算了整整一类基因组。
    """
    rows = {}
    bad = []
    for op in _ops(p):
        s = _space(p, op)
        tot = sum(s["prune_reasons"].values())
        rows[op] = {"sum": tot, "n_pruned": s["n_pruned"]}
        if tot != s["n_pruned"]:
            bad.append(op)
    return (not bad,
            f"原因之和与被过滤数一致：{not bad}；明细 {rows}",
            rows)


@_case("no_collapsed_duplicates_in_space")
def _i_dupes(p):
    """笛卡尔积**不应该**产生重复基因组（行式算子的压缩已在 `iter_raw` 里做了）。

    红的情况：某个基因在行式算子上没被压成常量 ⇒ 空间里出现同一配置的多份副本
    ⇒ 「空间 450 个基因组」这句话里有一半是重复的，而门槛只数个数。
    """
    bad = {op: _space(p, op)["n_collapsed_duplicates"] for op in _ops(p)
           if _space(p, op)["n_collapsed_duplicates"] != 0}
    return (not bad, f"折叠掉的重复：{bad or '（无）'}", bad)


@_case("gene_cardinality_matches_the_space")
def _i_card(p):
    """`space.json` 的逐基因取值个数 == `genes.json` 的 coverage。

    红的情况：两处各算一遍，其中一处的口径漂了（例如一处按合法空间算、
    另一处按原始空间算）—— 两个数都印在报告里，而它们互相矛盾。
    """
    bad = []
    for op in _ops(p):
        a = _space(p, op)["gene_cardinality"]
        b = _genes(p, op)["coverage"]["cardinality"]
        if a != b:
            bad.append({"op": op, "space": a, "genes": b})
    return (not bad, f"逐基因取值个数一致：{not bad}；不一致 {bad}", bad)


@_case("row_ops_hold_row_constant_genes_at_one")
def _i_row(p):
    """行式算子上 `tile_m` / `tile_k` 的取值个数必须是 1。

    红的情况：行式算子沿 N 维规约，K 维不存在 ⇒ 这两个基因没有可切的东西。
    若它们有多个取值，空间会被**无意义的副本**灌水（`n_genomes` 虚高），
    而门槛 `min_space_per_op` 会靠这些副本通过。
    """
    bad = []
    for op in _ops(p):
        if not bool(_cfg(p)["ops"][op]["row_wise"]):
            continue
        c = _space(p, op)["gene_cardinality"]
        for g in ("tile_m", "tile_k"):
            if c.get(g) != 1:
                bad.append({"op": op, "gene": g, "cardinality": c.get(g)})
    return (not bad,
            f"行式算子上 tile_m / tile_k 恒为 1：{not bad}；违反 {bad}",
            bad)


@_case("every_gene_value_survives_filtering")
def _i_reach(p):
    """**非行式**算子上，每个基因的每一个声明取值都至少出现在合法空间里。

    红的情况：过滤规则太紧，某个取值被整体砍掉 —— 于是「我们在搜这个基因」
    是一句假话，而空间大小看起来只是「小了一点」。
    """
    cfg = _cfg(p)
    bad = []
    for op in _ops(p):
        if bool(cfg["ops"][op]["row_wise"]):
            continue
        c = _space(p, op)["gene_cardinality"]
        for g in GENE_ORDER:
            want = len(cfg["genome"][g])
            if c.get(g, 0) != want:
                bad.append({"op": op, "gene": g, "declared": want,
                            "in_space": c.get(g, 0)})
    return (not bad,
            f"全部声明取值都活着：{not bad}；被整体砍掉的 {bad}",
            bad)


@_case("baseline_genome_lives_in_the_declared_gene_space")
def _i_baseline_in_space(p):
    """基线基因组的每个取值都必须在 `config.genome` 的取值表里。

    红的情况：基线取了一个空间外的值（例如 `tile_m=32` 而表里最大 16）——
    那样 `speedup` 的分母是一个**候选根本达不到的**配置，
    于是所有加速比都被系统性放大，而它看起来只是个偏大的百分比。
    """
    cfg = _cfg(p)
    bad = []
    for op in _ops(p):
        b = p["baseline.json"]["per_op"][op]["genome"]
        for g in GENE_ORDER:
            v = b[g]
            if g in ("tile_m", "tile_k") and bool(cfg["ops"][op]["row_wise"]):
                if v != 1:
                    bad.append({"op": op, "gene": g, "value": v,
                                "why": "行式算子上必须为 1"})
                continue
            if v not in list(cfg["genome"][g]):
                bad.append({"op": op, "gene": g, "value": v,
                            "why": "不在取值表里"})
    return (not bad, f"基线都在声明的基因空间里：{not bad}；越界 {bad}", bad)


@_case("probe_and_production_shapes_are_deliberately_different")
def _i_shapes(p):
    """**误差在探测形状上真跑，延迟在产出形状上解析算** —— 两个形状必须不同。

    红的情况：有人把两者设成一样，于是「延迟是解析值」这件事就变成
    「延迟是在那个形状上真跑出来的」—— 而本机没有 GPU，
    那句话是**把假设说成了测量**。两个形状相同 ⇒ 这个混淆无法被发现。
    """
    same = []
    for op in _ops(p):
        if list(_space(p, op)["probe_shape"]) == list(_space(p, op)["production_shape"]):
            same.append(op)
    return (not same,
            f"探测形状 ≠ 产出形状的算子 {len(_ops(p)) - len(same)}/{len(_ops(p))}；"
            f"相同的是 {same}",
            {"same_shape_ops": same})


# ===================================================================== 数值孪生
@_case("baseline_drift_is_exactly_zero")
def _i_base_zero(p):
    """基线对所有比较基准的漂移**恰好是 0.0** —— 它就是那个基准。

    红的情况：`drift_of` 的分母里混进了别的量（例如基线自己的 max 用错了
    数组），于是基线的漂移变成 1e-17 这样一个「几乎为零但不是零」的数。
    那会让「τ=0 那一级可行」这句话取决于浮点运气。
    """
    rows = {}
    for op in _ops(p):
        # τ=0 / reject 的 oracle 格子里必须含基线（它就是漂移为 0 的那个）
        c = _cell(p, op, "reject", 0.0)["oracle"]
        b = c["best_gate1_only"]
        rows[op] = {
            "n_feasible": c["n_feasible"],
            "baseline_label": p["baseline.json"]["per_op"][op]["label"],
            "best_gate1_label": (b or {}).get("label"),
            "best_gate1_drift": (b or {}).get("drift"),
        }
    bad = [op for op, r in rows.items() if r["n_feasible"] < 1]
    drift_bad = [op for op, r in rows.items()
                 if r["best_gate1_drift"] is not None
                 and r["best_gate1_drift"] > 1e-15]
    return (not bad and not drift_bad,
            f"τ=0 时每个算子都有可行 kernel：{not bad}；"
            f"最短延迟的那个的漂移 ≤ 1e-15：{not drift_bad}",
            {"rows": rows, "empty_ops": bad, "drift_ops": drift_bad})


@_case("drift_is_scale_invariant_and_zero_iff_bitwise_equal")
def _i_drift_props(p):
    """`drift_of` 的两条定义性性质：

    ① **尺度不变**：把两边同时乘 2，漂移不变（它按问题量级归一化）；
    ② **零 ⟺ 逐位相同**。

    红的情况：分母用成了 `max|got|`（那就成了「相对误差」，会随尺度变），
    或者用了逐元素相对误差（真值近零的元素会把读数放大到 1e-1，
    **盖过全部真实误差** —— 本项目第一版就是这么错的）。
    """
    cfg = _cfg(p)
    floor = float(cfg["numerics"]["error_floor"])
    bad = []
    n_checked = 0
    for op in _ops(p):
        ev = _new_evaluator(p, op)
        gens = [KernelGenome.from_dict(t["genome"])
                for t in _space(p, op)["sample_trials"]]
        if not gens:
            continue
        for g in gens[:2]:
            a = ev.num.get(g)
            b = list(ev.baseline_out)
            d1 = drift_of(a, b, floor)
            d2 = drift_of([2.0 * x for x in a], [2.0 * x for x in b], floor)
            n_checked += 1
            if abs(d1 - d2) > 1e-12 * max(1.0, abs(d1)):
                bad.append({"op": op, "why": "尺度不变性被破坏",
                            "d1": d1, "d2": d2})
            if (drift_of(a, a, floor) != 0.0):
                bad.append({"op": op, "why": "自己对自己的漂移不是 0"})
            if (d1 == 0.0) != (a == b):
                bad.append({"op": op, "why": "零 ⟺ 逐位相同 被破坏",
                            "d1": d1, "bitwise_equal": a == b})
    return (not bad,
            f"{n_checked} 对样本上两条性质都成立：{not bad}；违反 {bad[:4]}",
            {"n_checked": n_checked, "violations": bad[:4]})


@_case("kernel_output_is_deterministic")
def _i_determinism(p):
    """同一个基因组跑两遍，输出**逐位相同**。

    红的情况：内核里用了 `set` 迭代、字典序、或者未播种的随机数 ——
    那样「同一份输入两次 run 得到同一份产物」这条核对就失效了，
    而它看起来只是「有点不稳定」。
    """
    cfg = _cfg(p)
    bad = []
    n = 0
    for op in _ops(p):
        ev = _new_evaluator(p, op)
        for t in _space(p, op)["sample_trials"]:
            g = KernelGenome.from_dict(t["genome"])
            a = kernel_output(cfg, op, g, ev.num.ins)
            b = kernel_output(cfg, op, g, ev.num.ins)
            n += 1
            if a != b:
                bad.append({"op": op, "genome": t["label"]})
    return (not bad, f"{n} 个基因组各跑两遍：逐位相同 = {not bad}",
            {"n": n, "violations": bad[:4]})


@_case("cache_key_equality_implies_output_equality")
def _i_cache_sound(p):
    """**同键 ⇒ 同输出**（缓存正确性的唯一判据）。

    红的情况：缓存键漏了一个真的影响输出的成分。本项目第一版漏了 `mode`
    （`persistent` 与 `default` 的归约方式不同）⇒ `persistent` 的 kernel
    拿到了 `default` 的输出。**缓存命中率很漂亮，结果是错的。**

    ⚠️ 这条用例**天然容易退化成恒真**（若所有键都互不相同）。
    所以它必须同时报 `n_same_key_pairs` 并要求 ≥ 1 —— 否则
    「同键 ⇒ 同输出」是一句没有检验过任何东西的话。
    """
    cfg = _cfg(p)
    bad = []
    same_key_pairs = 0
    n = 0
    for op in _ops(p):
        ev = _new_evaluator(p, op)
        entries: list[tuple[Any, list[float], str]] = []

        def add(g: KernelGenome, label: str, variant: str | None = None) -> None:
            bits = SIGNIFICANT_BITS[g.accum_dtype]
            if variant == "no_quantize_declared_fp32":
                bits = SIGNIFICANT_BITS["bf16"]
            k = cache_key(op, g, bits, variant)
            entries.append((k, kernel_output(cfg, op, g, ev.num.ins,
                                             variant=variant), label))

        # ① 抽样：合法路径
        for t in _space(p, op)["sample_trials"]:
            add(KernelGenome.from_dict(t["genome"]), t["label"])
        # ② 变体路径（键里额外含 tile_m / tile_n）
        for name in ILLEGAL_VARIANTS[op]:
            add(ev.baseline, f"{op}/{name}", variant=name)
        # ③ **必须造出「应该同键」的记录**，否则「同键 ⇒ 同输出」是空洞的。
        #    改一个**不影响加法树**的基因 ⇒ 键与输出都**必须**不变。
        #    （这正是缓存命中率 0.95 的来源，也是那条用例唯一有牙齿的地方。）
        base = ev.baseline
        for gene, vals in (("memory_layout", cfg["genome"]["memory_layout"]),
                           ("vector_width", cfg["genome"]["vector_width"]),
                           ("num_stages", cfg["genome"]["num_stages"])):
            for v in vals:
                if v == getattr(base, gene):
                    continue
                d = base.to_dict()
                d[gene] = v
                add(KernelGenome.from_dict(d), f"{op}/{gene}={v}")

        n += len(entries)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i][0] != entries[j][0]:
                    continue
                same_key_pairs += 1
                if entries[i][1] != entries[j][1]:
                    bad.append({"op": op, "a": entries[i][2], "b": entries[j][2]})
    ok = not bad and same_key_pairs >= 1
    return (ok,
            f"{n} 条记录上出现 {same_key_pairs} 对**同键**记录，"
            f"其中输出不同的 {len(bad)} 对；"
            + ("同键 ⇒ 同输出 成立" if not bad else f"**违反**：{bad[:4]}")
            + ("" if same_key_pairs >= 1
               else "；⚠️ **没有任何同键对 ⇒ 这条用例是空洞的**"),
            {"n_records": n, "n_same_key_pairs": same_key_pairs,
             "violations": bad[:4]})


def t_label(op: str, name: str) -> str:
    return f"{op}/{name}"


@_case("cache_key_covers_every_gene_the_kernel_reads")
def _i_cache_ast(p):
    """AST 扫 `ops.py`：内核读过的每个 `gen.<基因>` 都必须在缓存键的成分里。

    红的情况：往数值函数里加一个新基因、忘了加进 `CACHE_KEY_COMPONENTS` ——
    于是缓存把两个不同的 kernel 当成同一个。**这是「同键 ⇒ 同输出」那条
    用例的静态补充**：那条是抽样核，这条是全量扫源码。
    """
    from pathlib import Path
    src = Path(__file__).resolve().parent / "ops.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    read: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ("gen", "g")
                and node.attr in GENE_ORDER):
            read.add(node.attr)
    # `accum_dtype`（→ bits）与 `scheduling_strategy`（→ mode）在 cache_key 里
    # 是**被换算成别的量**后进键的，所以它们不需要作为裸成分出现。
    handled_specially = {"accum_dtype", "scheduling_strategy"}
    covered = set().union(*CACHE_KEY_COMPONENTS.values()) \
        | set(VARIANT_KEY_COMPONENTS) | handled_specially
    missing = sorted(read - covered)
    return (not missing,
            f"内核读了 {sorted(read)}；键的成分覆盖了 {sorted(covered)}；"
            + ("全覆盖" if not missing else f"**没覆盖**：{missing}"),
            {"read_by_kernel": sorted(read), "missing_from_key": missing})


@_case("numerics_cache_hit_rate_is_high")
def _i_cache_rate(p):
    """数值缓存命中率 > 0.5。

    红的情况：缓存键里混进了**不影响输出**的基因（本项目第一版把 `tile_m`
    放进了合法路径的键）—— 于是键空间从 36 膨胀到 384，
    每一次都是「注定命中的未命中」。**这不会算错任何东西**，
    只让整条流水线慢十倍 —— 而慢十倍与「空间变大了一点」在输出里无法区分。
    """
    rates = {}
    bad = []
    for op in _ops(p):
        d = _space(p, op)["evaluator"]["numerics"]
        n = d["cache_hits"] + d["unique_outputs"]
        rates[op] = round((d["cache_hits"] / n) if n else 0.0, 6)
        if n and rates[op] <= 0.5:
            bad.append(op)
    return (not bad, f"命中率 {rates}（要求全部 > 0.5）；不达标 {bad}", rates)


@_case("effective_bits_is_ordered_by_declared_precision")
def _i_bits_order(p):
    """三个精度档的**实测**有效位宽区间必须严格分开：
    `fp32 > tf32 > bf16`（按区间不重叠）。

    红的情况：`effective_bits` 的分母（同算法 fp64）用错了对象，
    或者输入量化根本没生效 —— 三个档会挤在一起，
    而「有效位宽」这条读数就退化成「声明位宽」的复述。
    """
    bad = []
    detail = {}
    for op in _ops(p):
        e = _space(p, op)["effective_bits_by_dtype"]
        detail[op] = {k: [v["min"], v["max"]] for k, v in e.items()}
        for hi, lo in (("fp32", "tf32"), ("tf32", "bf16")):
            if hi not in e or lo not in e:
                bad.append({"op": op, "why": f"{hi} 或 {lo} 不在空间里"})
                continue
            if not (e[hi]["min"] > e[lo]["max"]):
                bad.append({"op": op, "pair": f"{hi}>{lo}",
                            "hi_min": e[hi]["min"], "lo_max": e[lo]["max"]})
    return (not bad,
            f"三档区间两两不重叠的算子 {len(_ops(p)) - len({b['op'] for b in bad})}"
            f"/{len(_ops(p))}；违反 {bad}",
            {"ranges": detail, "violations": bad})


@_case("effective_bits_never_exceeds_declared_bits")
def _i_bits_cap(p):
    """合法 kernel 的实测有效位宽 ≤ 声明位宽 + 0.5。

    红的情况：`effective_bits` 把「与 fp32 参考的偏差」当成了分母 ——
    那样 fp32 kernel 的偏差里混进了算法差异，而有效位宽会**超过** 24 位。
    一个「比声明还准」的读数不可能来自精度模型，只可能来自口径错位。
    """
    bad = []
    worst = None
    for op in _ops(p):
        for t in _space(p, op)["sample_trials"]:
            excess = t["effective_bits"] - t["declared_bits"]
            if worst is None or excess > worst[0]:
                worst = (excess, f"{op}/{t['label']}")
            if excess > 0.5:
                bad.append({"op": op, "label": t["label"],
                            "effective_bits": t["effective_bits"],
                            "declared_bits": t["declared_bits"]})
    return (not bad,
            f"最大超出 {None if worst is None else round(worst[0], 6)} 位"
            f"（{None if worst is None else worst[1]}）；超标的 {bad}",
            {"worst_excess": None if worst is None else round(worst[0], 6),
             "violations": bad})


@_case("bits_bias_is_a_pure_function_of_the_step_count")
def _i_bias(p):
    """产物里的 `bits_bias` 必须能由 `accum_steps(cfg, op, genome)` 重算出来。

    红的情况：`accum_steps` 改了（例如在线归约的步数从 dim 改成 log2 dim），
    但产物里的偏置是旧的 —— 于是「有效位宽 ≈ 声明 − 偏置」这条期望
    会拿一个**算不出来的偏置**去核实测值，而它永远差那么一点。
    """
    cfg = _cfg(p)
    bad = []
    n = 0
    for op in _ops(p):
        for t in _space(p, op)["sample_trials"]:
            g = KernelGenome.from_dict(t["genome"])
            want = expected_bits_bias(accum_steps(cfg, op, g))
            n += 1
            if abs(want - t["bits_bias"]) > 1e-9:
                bad.append({"op": op, "label": t["label"],
                            "recorded": t["bits_bias"], "recomputed": want})
    return (not bad, f"{n} 个样本上偏置可重算：{not bad}；不一致 {bad[:4]}",
            {"n": n, "violations": bad[:4]})


def _inert_ok(p, gene: str) -> tuple[bool, dict[str, Any]]:
    rows = {}
    bad = []
    for op in _ops(p):
        r = next((x for x in _genes(p, op)["rows"] if x["gene"] == gene), None)
        if r is None:
            bad.append({"op": op, "why": "基因不在表里"})
            continue
        rows[op] = {"n_pairs": r["n_pairs"],
                    "max_drift_delta": r["max_drift_delta"],
                    "numerically_inert": r["numerically_inert"]}
        if r["n_pairs"] == 0:
            bad.append({"op": op, "why": "**没有任何反事实配对**（没测到，不是没影响）"})
        elif r["max_drift_delta"] != 0.0:
            bad.append({"op": op, "why": "漂移差不为零",
                        "max_drift_delta": r["max_drift_delta"]})
    return (not bad, rows)


@_case("memory_layout_is_numerically_inert_everywhere")
def _i_layout(p):
    """`memory_layout` 对误差的影响**恰好为零**（在每一个算子上）。

    红的情况：有人在数值函数里读了 `memory_layout`（例如「swizzled 用不同的
    累加顺序」）—— 那是**建模错误**：访存布局不改变加法树。
    它会让「布局是免费的」这条结论消失，而漂移只变一点点。
    """
    ok, rows = _inert_ok(p, "memory_layout")
    return (ok, f"逐算子：{rows}；结论 {ok}", rows)


@_case("vector_width_is_numerically_inert_everywhere")
def _i_vec(p):
    """`vector_width` 对误差的影响**恰好为零**（它只改访存宽度）。"""
    ok, rows = _inert_ok(p, "vector_width")
    return (ok, f"逐算子：{rows}；结论 {ok}", rows)


@_case("num_stages_is_numerically_inert_everywhere")
def _i_stages(p):
    """`num_stages` 对误差的影响**恰好为零**（软件流水线不改加法树）。"""
    ok, rows = _inert_ok(p, "num_stages")
    return (ok, f"逐算子：{rows}；结论 {ok}", rows)


@_case("accum_dtype_dominates_the_error")
def _i_dtype_dominates(p):
    """`accum_dtype` 的漂移效应必须**压倒性**：bf16 的最大漂移
    ≥ 10 × fp32 的最大漂移（逐算子）。

    红的情况：精度轴只接进了成本模型、没接进数值模型（或接反了）——
    那样「放宽容差买到速度」这条结论的来源就说不清了。
    """
    rows = {}
    bad = []
    for op in _ops(p):
        e = _space(p, op)["effective_bits_by_dtype"]
        d = {}
        for t in _space(p, op)["sample_trials"]:
            dt = t["genome"]["accum_dtype"]
            d.setdefault(dt, []).append(t["drift"])
        if not d.get("bf16") or not d.get("fp32"):
            bad.append({"op": op, "why": "缺 fp32 或 bf16 样本"})
            continue
        ratio = max(d["bf16"]) / max(max(d["fp32"]), 1e-300)
        rows[op] = {"max_drift_bf16": max(d["bf16"]),
                    "max_drift_fp32": max(d["fp32"]),
                    "ratio": round(ratio, 6),
                    "effective_bits_range": {k: [v["min"], v["max"]]
                                             for k, v in e.items()}}
        if ratio < 10.0:
            bad.append({"op": op, "ratio": ratio})
    return (not bad,
            f"bf16/fp32 漂移比 {[(o, r['ratio']) for o, r in rows.items()]}"
            f"（要求全部 ≥ 10）；不达标 {bad}",
            rows)


# ===================================================================== 两层门
@_case("deliverable_is_the_conjunction_of_the_two_gates")
def _i_conj(p):
    """`deliverable ⟺ feasible ∧ bits_ok`，逐点核。

    红的情况：交付判据被写成「或」或者只看门 1 —— 于是前沿里会混进
    「声称 fp32、实际 8 位」的 kernel，而它看起来只是一个又快又准的点。
    """
    bad = []
    n = 0
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                for t in [cell["best"], cell["best_gate1_only"], cell["best_score"]] \
                        + cell["frontier"]:
                    if t is None:
                        continue
                    n += 1
                    if bool(t["deliverable"]) != (bool(t["feasible"])
                                                 and bool(t["bits_ok"])):
                        bad.append({"op": op, "cell": f"{name}@{key}",
                                    "genome": t["label"], "feasible": t["feasible"],
                                    "bits_ok": t["bits_ok"],
                                    "deliverable": t["deliverable"]})
    return (not bad, f"{n} 个点上合取成立：{not bad}；违反 {bad[:4]}",
            {"n": n, "violations": bad[:4]})


@_case("gate2_verdict_does_not_depend_on_the_tolerance")
def _i_gate2_fixed(p):
    """门 2（`bits_ok`）**与 τ 无关**：同一个基因组在阶梯的每一级上结论相同。

    红的情况：门 2 被实现成 `drift ≤ τ ∧ 声明检查`（那就成了门 1 的一部分）——
    于是「门 1 的宽松度不该影响诚实性判定」这条设计意图消失，
    而报告里两栏的读数看起来仍然正常。
    """
    seen: dict[tuple[str, str], set[bool]] = {}
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                for t in [cell["best"], cell["best_gate1_only"]] + cell["frontier"]:
                    if t is None:
                        continue
                    seen.setdefault((op, t["genome_id"]), set()).add(bool(t["bits_ok"]))
    bad = [{"op": o, "genome_id": g} for (o, g), v in seen.items() if len(v) > 1]
    return (not bad,
            f"{len(seen)} 个基因组上 bits_ok 结论唯一：{not bad}；矛盾的 {bad[:4]}",
            {"n_genomes": len(seen), "violations": bad[:4]})


@_case("gate1_acceptance_is_monotone_in_the_tolerance")
def _i_gate1_mono(p):
    """对每个非法变体，`accepted_at` 必须是 τ 的**单调不减**布尔序列。

    红的情况：`accepted_at` 的键与容差对不上（例如一处用 `str(t)`、
    另一处用 `f"{t:g}"`）⇒ 某一格读到的是「没查到 ⇒ False」，
    于是序列里冒出一个假的下降，而它看起来只是「这一档没接受」。
    """
    tols = _tolerances(p)
    bad = []
    for op in _ops(p):
        for r in _illegal(p, op):
            seq = [_at(r["accepted_at"], t) for t in tols]
            if seq != sorted(seq):
                bad.append({"op": op, "variant": r["variant"], "sequence": seq})
            # τ 最小那一级（0.0）必须**全部被拒**：非法变体的漂移 > 0
            if seq and seq[0]:
                bad.append({"op": op, "variant": r["variant"],
                            "why": "在 τ=0 就被接受了（漂移应当 > 0）"})
    return (not bad, f"单调序列全部成立：{not bad}；违反 {bad[:4]}",
            {"n_variants": sum(len(_illegal(p, op)) for op in _ops(p)),
             "violations": bad[:4]})


@_case("gate2_never_rejects_a_legal_kernel")
def _i_gate2_clean(p):
    """门 2 **不误伤**任何合法 kernel（合法空间里被门 2 拒掉的必须是 0 个）。

    红的情况：`precision_slack_bits` 标定得太紧 —— 于是门 2 开始拒合法 kernel，
    而「门 2 是诚实的检查器」这句话就变成了「门 2 是个过严的门槛」。
    """
    bad = {op: _space(p, op)["n_legal_rejected_by_gate2"] for op in _ops(p)
           if _space(p, op)["n_legal_rejected_by_gate2"] != 0}
    return (not bad, f"被门 2 误伤的合法 kernel：{bad or '（无）'}", bad)


@_case("gate2_separates_honest_bf16_from_fake_fp32")
def _i_gate2_sep(p):
    """**本项目最重要的一条**：头条对照里，门 1 对两行结论相同、
    而门 2 把它们分开。

    红的情况：门 2 的实现与声明**无关**（例如它拿 drift 去比一个固定阈值）——
    那样它对两行给出同一个结论，而报告里那张「两行漂移同量级」的表
    看起来完全正常。**这张表存在的全部意义就是这一条断言。**

    ⚠️ 对照必须做在 `nearest_honest_bf16`（漂移最接近假 fp32 的那个诚实 bf16）上。
    第一版拿「漂移最大的诚实 bf16」比，在 matmul 上红了 ——
    真因是**代表选错了**，不是结论错。
    """
    rows = {}
    bad = []
    for op in _ops(p):
        h = _adv(p, op).get("headline", {})
        if not h.get("available"):
            bad.append({"op": op, "why": "对照表不可用", "reason": h.get("why")})
            continue
        n = h["nearest_honest_bf16"]
        f = h["fake_fp32"]
        rows[op] = {
            "honest_label": n["label"], "fake_label": f["label"],
            "honest_drift": n["drift"], "fake_drift": f["drift"],
            "drift_ratio": n.get("drift_ratio"),
            "gate1_honest": n["gate1_feasible_at_1e-2"],
            "gate1_fake": f["gate1_feasible_at_1e-2"],
            "gate2_honest": n["gate2_bits_ok"],
            "gate2_fake": f["gate2_bits_ok"],
        }
        if not h.get("gate1_same_verdict"):
            bad.append({"op": op, "why": "门 1 对最难区分的那一对结论**不同**"})
        if not n["gate2_bits_ok"]:
            bad.append({"op": op, "why": "门 2 把诚实的 bf16 也拒了"})
        if f["gate2_bits_ok"]:
            bad.append({"op": op, "why": "门 2 **放行了**假装 fp32 —— 它没起作用"})
        if not h.get("gate2_separates"):
            bad.append({"op": op, "why": "gate2_separates 标志与逐项读数不一致"})
    return (not bad, f"门 2 是唯一分开两行的判据：{not bad}；违反 {bad}", rows)


@_case("fake_fp32_sits_inside_the_honest_bf16_error_range")
def _i_ratio(p):
    """假 fp32 的漂移必须落在**全部诚实 bf16** 的漂移区间**之内**。

    这一条是**可被推翻的**：若那个变体实际按 2 位（而不是 bf16 的 8 位）跑，
    它的漂移（≈2^−2 = 0.25）会远大于任何诚实 bf16（≈9e−3），
    于是它落在区间之外 —— **「假 fp32 混在诚实 bf16 里看不出来」这句话当场失效**。
    那正是「门 1 分不开它们」这个论断的前提。

    同时报出「最难区分那一对」的漂移比（实测在 layernorm 上恰好 1.0 ——
    两者**逐位相同**）。
    """
    rows = {}
    bad = []
    for op in _ops(p):
        h = _adv(p, op).get("headline", {})
        if not h.get("available"):
            bad.append({"op": op, "why": "对照表不可用"})
            continue
        rng = h["honest_bf16_range"]
        n = h["nearest_honest_bf16"]
        rows[op] = {"fake_drift": h["fake_fp32"]["drift"],
                    "honest_range": [rng["min_drift"], rng["max_drift"]],
                    "inside": h["fake_inside_honest_range"],
                    "nearest_drift_ratio": n.get("drift_ratio"),
                    "bit_identical_drift": n.get("bit_identical_drift")}
        if not h["fake_inside_honest_range"]:
            bad.append({"op": op, "why": "假 fp32 落在诚实 bf16 的漂移区间之外",
                        "fake_drift": h["fake_fp32"]["drift"],
                        "range": [rng["min_drift"], rng["max_drift"]]})
        if not rng["n"]:
            bad.append({"op": op, "why": "诚实 bf16 的样本数为 0"})
    if not rows:
        return False, "对照表一个都不可用 —— 这条用例没法核", {}
    return (not bad, f"假 fp32 落在诚实 bf16 的漂移区间内：{not bad}；违反 {bad}",
            rows)


@_case("illegal_variants_all_declare_fp32")
def _i_declare(p):
    """**全部**具名非法变体的声明精度必须是 `fp32`。

    红的情况：`no_quantize_declared_fp32` 被填成 `bf16`（本项目第一版就是这样）——
    于是它的门 2 头寸 ≈ 0 ⇒ **门 2 放行它**，而它正是「声明与实现不符」的样本。
    那个错误让头条对照表印出「门 2 也分不开」，**与结论相反，且完全看不出来**。
    """
    bad = []
    for op in _ops(p):
        for r in _adv(p, op)["variants"]:
            if r["declared_dtype"] != "fp32":
                bad.append({"op": op, "variant": r["variant"],
                            "declared_dtype": r["declared_dtype"]})
    return (not bad,
            f"{sum(len(_adv(p, op)['variants']) for op in _ops(p))} 个非法变体"
            f"全部声称 fp32：{not bad}；例外 {bad}",
            bad)


@_case("illegal_variants_all_have_nonzero_drift")
def _i_illegal_drift(p):
    """每个非法变体的漂移必须 **> 0**（它真的造成了偏差）。

    红的情况：变体没被执行到 —— 例如基线分块恰好整除探测形状，
    于是 `drop_tail` 的漂移**精确等于 0**，它看起来像「这个变体没生效」，
    其实是**用例根本没被执行到**。本项目第一版就是这样。
    """
    bad = []
    for op in _ops(p):
        for r in _adv(p, op)["variants"]:
            if not (r["drift"] > 0.0):
                bad.append({"op": op, "variant": r["variant"], "drift": r["drift"]})
    return (not bad, f"漂移 > 0 的非法变体：{not bad}；零漂移的 {bad}", bad)


@_case("illegal_variant_count_matches_the_variant_table")
def _i_variant_count(p):
    """每算子的变体数 == 尺子档数 + 具名 bug 数。

    红的情况：某个算子的具名变体表里少了一项（例如
    `no_quantize_declared_fp32` 只挂在 matmul 上）—— 于是另外两个算子的
    头条对照静默返回 `available=False`，**报告里那一节直接消失，不报错**。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        a = _adv(p, op)
        want = len(_cfg(p)["adversarial"]["mantissa_bits"]) + len(ILLEGAL_VARIANTS[op])
        rows[op] = {"n_variants": a["n_variants"], "expected": want,
                    "n_rulers": a["n_rulers"],
                    "named": sorted(set(ILLEGAL_VARIANTS[op]))}
        if a["n_variants"] != want:
            bad.append({"op": op, "n_variants": a["n_variants"], "expected": want})
        if set(ILLEGAL_VARIANTS[op]) - {r["variant"] for r in a["variants"]}:
            bad.append({"op": op, "why": "具名变体有缺失"})
    return (not bad, f"变体数与表一致：{not bad}；不一致 {bad}", rows)


@_case("illegal_variants_all_carry_an_explanatory_note")
def _i_variant_note(p):
    """每个非法变体都有一段说明它对应哪种真实实现错误的 `note`。

    红的情况：新增一个变体忘了写 note —— 于是报告里那一行只有名字，
    读者无法知道它是「什么 bug」，而**一个没有解释的变体与一个装饰性的
    扰动在输出里长得一模一样**。
    """
    bad = []
    for op in _ops(p):
        for r in _adv(p, op)["variants"]:
            if not str(r.get("note", "")).strip():
                bad.append({"op": op, "variant": r["variant"]})
    return (not bad, f"全部变体都有说明：{not bad}；缺说明的 {bad}", bad)


@_case("safe_window_bounds_come_from_the_raw_drifts")
def _i_window(p):
    """安全窗口的两端必须**从原始漂移取**：
    下界 = 合法空间漂移的最大值；上界 = 全部非法变体漂移的最小值。

    红的情况：下界拿**基线**比（漂移恒为 0）⇒ 窗口看起来从 0 开始 ——
    而 0 是「基线自己」，不是「合法实现能达到的最差精度」。
    于是「窗口非空」这句话变成了一句关于基线的、没有信息量的话。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        w = _adv(p, op)["safe_window"]
        ill = [r["drift"] for r in _illegal(p, op)]
        want_hi = min(ill) if ill else None
        rows[op] = {"lo": w["lo"], "hi": w["hi"], "illegal_min": want_hi,
                    "nonempty": w["nonempty"]}
        if want_hi is not None and abs(w["hi"] - want_hi) > 1e-15:
            bad.append({"op": op, "why": "上界 ≠ 非法变体最小漂移",
                        "hi": w["hi"], "want": want_hi})
        if w["lo"] <= 0.0:
            bad.append({"op": op, "why": "下界 ≤ 0 —— 多半是拿基线比的", "lo": w["lo"]})
    return (not bad, f"窗口两端都从原始漂移取：{not bad}；违反 {bad}", rows)


@_case("ruler_grid_spans_the_cliff")
def _i_ruler(p):
    """尺子网格必须**横跨**「合法实现能达到的最差精度」那条线（两侧都非空）。

    红的情况：网格全在线的同一侧 ⇒ 「合法低精度」与「偷偷降精度」在误差上
    重叠的那一段**没有被测量到**，而那张表看起来完全正常。
    （本项目第一版把断言写成「至少一档被接受、至少一档不被接受」——
    阶梯顶端放到 1.0 ⇒ 必然全被接受 ⇒ 那条断言**必然触发**，
    而真因是「阶梯太长」，不是崖没被测量到。）
    """
    bad = []
    rows = {}
    for op in _ops(p):
        a = _adv(p, op)
        rows[op] = {"spans": a["ruler_grid_spans_cliff"],
                    "below": a["n_rulers_below_legal_max"],
                    "above": a["n_rulers_above_legal_max"],
                    "legal_max": a["safe_window"]["legal_max_drift"]}
        if not a["ruler_grid_spans_cliff"]:
            bad.append({"op": op, "why": "网格没有横跨崖",
                        "below": a["n_rulers_below_legal_max"],
                        "above": a["n_rulers_above_legal_max"]})
    return (not bad, f"横跨崖的算子 {len(_ops(p)) - len(bad)}/{len(_ops(p))}；违反 {bad}",
            rows)


@_case("accepted_at_keys_are_exactly_the_tolerances")
def _i_keys(p):
    """`accepted_at` 的键集合必须**恰好**是容差集合（`str(float(t))`）。

    红的情况：键被写成 `f"{t:g}"` ⇒ `0.0` 那一级的键是 `'0'` 而不是 `'0.0'`，
    于是「τ=0 时非法变体全被拒」这句话里 **τ=0 那一格根本没被检查**。
    **一个永远查不到的键，与「查到了且通过」在输出里长得一模一样。**
    """
    tols = {str(float(t)) for t in _tolerances(p)}
    bad = []
    for op in _ops(p):
        for r in _adv(p, op)["variants"]:
            got = set(r["accepted_at"])
            if got != tols:
                bad.append({"op": op, "variant": r["variant"],
                            "missing": sorted(tols - got),
                            "extra": sorted(got - tols)})
    return (not bad,
            f"键集合与容差集合一致（{len(tols)} 档）：{not bad}；违反 {bad[:4]}",
            {"expected_keys": sorted(tols), "violations": bad[:4]})


# ===================================================================== 阶梯与前沿
@_case("reference_is_shared_by_every_cell_of_an_op")
def _i_ref(p):
    """同一个算子的**全部**格子（10 级 τ × 2 策略）共用同一个参考点。

    红的情况：按级取参考点（`ref = min(该级可行域) − pad`）会让
    「放宽容差」同时移动参考点 ⇒ 超体积凭空上涨。
    `L6-graphopt` 实测涨了 68%，**而那 68% 全是假象**。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        refs = {tuple(r["reference"]) for r in _ladder(p, op).values()}
        rows[op] = sorted(refs)
        if len(refs) != 1:
            bad.append({"op": op, "n_distinct_references": len(refs)})
    return (not bad, f"每算子唯一参考点：{not bad}；违反 {bad}", rows)


@_case("hypervolume_is_monotone_in_the_tolerance")
def _i_hv_mono(p):
    """τ 放松 ⇒ 可行域扩张 ⇒ 前沿只增不减 ⇒ 超体积不减（**数学必然**）。

    红了就是实现错了 —— 不是「这次没达标」。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        for strat in ("reject", "penalty"):
            hv = [r["cells"]["oracle"]["hypervolume"]
                  for _t, r in sorted(_ladder(p, op).items(),
                                      key=lambda kv: kv[1]["tolerance"])
                  if r["strategy"] == strat]
            rows[f"{op}/{strat}"] = [round(x, 9) for x in hv]
            if any(b < a - 1e-12 for a, b in zip(hv, hv[1:])):
                bad.append({"op": op, "strategy": strat})
    return (not bad, f"超体积单调不减：{not bad}；违反 {bad}", rows)


@_case("deliverable_count_is_monotone_in_the_tolerance")
def _i_deliv_mono(p):
    """交付集大小（过两道门）随 τ 单调不减。

    红的情况：门 2 被写成了 τ 的函数 —— 那样交付集可能**收缩**，
    而「阶梯」这个词就不再成立（放松约束不该让可选项变少）。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        for strat in ("reject", "penalty"):
            n = [r["cells"]["oracle"]["n_deliverable"]
                 for _t, r in sorted(_ladder(p, op).items(),
                                     key=lambda kv: kv[1]["tolerance"])
                 if r["strategy"] == strat]
            rows[f"{op}/{strat}"] = n
            if any(b < a for a, b in zip(n, n[1:])):
                bad.append({"op": op, "strategy": strat, "sequence": n})
    return (not bad, f"交付集大小单调不减：{not bad}；违反 {bad}", rows)


@_case("frontier_points_are_pairwise_non_dominated")
def _i_front_nd(p):
    """前沿里**没有任何**点被另一个前沿点支配。

    红的情况：`non_dominated_indices` 的实现漏了一对（例如排序扫描的边界写错）——
    于是「前沿」里混进了被支配的点，而它看起来只是「前沿大了一点」。
    """
    bad = []
    n = 0
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                pts = [t["objectives"] for t in cell["frontier"]]
                n += len(pts)
                for i in range(len(pts)):
                    for j in range(len(pts)):
                        if i != j and dominates(pts[j], pts[i]):
                            bad.append({"op": op, "cell": f"{name}@{key}",
                                        "i": cell["frontier"][i]["label"],
                                        "j": cell["frontier"][j]["label"]})
    return (not bad, f"{n} 个前沿点上两两不支配：{not bad}；违反 {bad[:4]}",
            {"n_points": n, "violations": bad[:4]})


@_case("frontier_is_a_subset_of_the_deliverable_set")
def _i_front_sub(p):
    """前沿 ⊆ 交付集（过两道门）。

    红的情况：前沿建在 `feasible` 上而不是 `deliverable` 上 ——
    于是前沿里出现「声称 fp32、实际 8 位」的 kernel，
    而它看起来只是一个「又快又准」的点。
    """
    bad = []
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                for t in cell["frontier"]:
                    if not t["deliverable"]:
                        bad.append({"op": op, "cell": f"{name}@{key}",
                                    "genome": t["label"],
                                    "feasible": t["feasible"],
                                    "bits_ok": t["bits_ok"]})
    return (not bad, f"前沿全部过两道门：{not bad}；违反 {bad[:4]}", bad[:8])


@_case("hypervolume_matches_the_frontier_recomputed")
def _i_hv_recompute(p):
    """从产物里的前沿点**重算**超体积 == 记录的超体积。

    红的情况：超体积与前沿是用两套不同的点集算的（例如超体积用了
    `feasible`、前沿用了 `deliverable`）—— 两个数都印在报告里，而它们互相矛盾。

    ⚠️ 容差是 1e-6 而不是 1e-12：前沿点的 `speedup` / `effective_bits` 在产物里
    被舍入到 9 位小数，而超体积是「(x−ref_x)·(y−ref_y)」的**乘积** ——
    舍入误差会被放大到约 1e-8 量级。**把容差取得比舍入误差还小，
    这条用例就变成了一条永远红的用例，而永远红与永远绿一样没有信息量。**
    """
    bad = []
    rows = {}
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            ref = rung["reference"]
            for name, cell in rung["cells"].items():
                pts = [[t["speedup"], t["effective_bits"]]
                       for t in cell["frontier"]]
                got = hypervolume_2d(pts, ref)
                if abs(got - cell["hypervolume"]) > 1e-6:
                    bad.append({"op": op, "cell": f"{name}@{key}",
                                "recorded": cell["hypervolume"],
                                "recomputed": got})
        rows[op] = len(_ladder(p, op))
    return (not bad, f"重算一致：{not bad}；不一致 {bad[:4]}",
            {"n_cells_per_op": rows, "violations": bad[:4]})


@_case("each_cell_reports_both_gates_consistently")
def _i_cell_consistency(p):
    """每个格子的两个「最好」必须自洽：

    - `n_deliverable ≤ n_feasible`（交付集 ⊆ 可行集）；
    - `n_deliverable > 0 ⟺ best 非空`；
    - `n_feasible > 0 ⟺ best_gate1_only 非空`；
    - **`best_gate1_only` 的延迟 ≤ `best` 的延迟**（可行集更大 ⇒ 最优不会更差）。

    最后一条是本项目两门设计的交叉核对：只过门 1 的最优**必须**不慢于
    过两道门的最优 —— 否则说明两个「最好」取自不同的集合口径。
    """
    bad = []
    n = 0
    for op in _ops(p):
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                n += 1
                tag = f"{op}/{name}@{key}"
                if cell["n_deliverable"] > cell["n_feasible"]:
                    bad.append({"cell": tag, "why": "交付集比可行集大"})
                if (cell["n_deliverable"] > 0) != (cell["best"] is not None):
                    bad.append({"cell": tag, "why": "best 与 n_deliverable 不一致"})
                if (cell["n_feasible"] > 0) != (cell["best_gate1_only"] is not None):
                    bad.append({"cell": tag,
                                "why": "best_gate1_only 与 n_feasible 不一致"})
                a, b = cell["best"], cell["best_gate1_only"]
                if a is not None and b is not None and b["latency_us"] > a["latency_us"] + 1e-9:
                    bad.append({"cell": tag, "why": "只过门 1 的最优比过两门的更慢",
                                "gate1_latency_us": b["latency_us"],
                                "both_gates_latency_us": a["latency_us"]})
    return (not bad, f"{n} 个格子的两门读数自洽：{not bad}；违反 {bad[:4]}",
            {"n_cells": n, "violations": bad[:4]})


@_case("best_answer_is_deliverable_and_honest")
def _i_best_answer(p):
    """交出去的答案必须**过两道门**，且它的 speedup 能从延迟重算出来。

    红的情况：`best_answer` 取自 `best_score`（那在 penalty 策略下可能不可行）——
    于是头条数字来自一个**不可行**的点，而它看起来只是一个漂亮的百分比。
    """
    cfg = _cfg(p)
    bad = []
    rows = {}
    for op in _ops(p):
        b = p["ladder.json"]["best_answer"].get(op)
        if b is None:
            bad.append({"op": op, "why": "τ=0/reject 下没有交出答案"})
            continue
        base_lat = float(p["baseline.json"]["per_op"][op]["metrics"]["latency_us"])
        want = base_lat / float(b["latency_us"])
        rows[op] = {"label": b["label"], "feasible": b["feasible"],
                    "bits_ok": b["bits_ok"], "drift": b["drift"],
                    "speedup_recorded": b["objectives"][0],
                    "speedup_recomputed": want}
        if not b["feasible"] or not b["bits_ok"]:
            bad.append({"op": op, "why": "交出的答案没过两道门"})
        if abs(want - b["objectives"][0]) > 1e-6 * max(1.0, abs(want)):
            bad.append({"op": op, "why": "speedup 与延迟之比对不上",
                        "recorded": b["objectives"][0], "recomputed": want})
        if b["drift"] > float(cfg["ladder"]["tolerances"][0]) + 1e-15:
            bad.append({"op": op, "why": "τ=0 交出的答案漂移不为 0",
                        "drift": b["drift"]})
    return (not bad, f"头条答案全部过两门且口径一致：{not bad}；违反 {bad}", rows)


@_case("search_budget_is_never_exceeded")
def _i_budget(p):
    """每个格子的 `budget_used` ≤ 该搜索器的预算；oracle 的预算 = 空间大小。

    红的情况：`Visits.used` 把「自己重复提案」也算进去（或反过来漏算）——
    于是「提案 400 次、预算用了 0」这种读数会出现，而它看起来只是一个很省的搜索器。
    """
    budget = int(_cfg(p)["search"]["budget_per_searcher"])
    bad = []
    rows = {}
    for op in _ops(p):
        n = _space(p, op)["n_genomes"]
        for key, rung in _ladder(p, op).items():
            for name, cell in rung["cells"].items():
                limit = n if name == "oracle" else budget
                if cell["budget_used"] > limit:
                    bad.append({"op": op, "cell": f"{name}@{key}",
                                "used": cell["budget_used"], "limit": limit})
                if name == "oracle" and cell["n_evaluated_unique"] != n:
                    bad.append({"op": op, "cell": key,
                                "why": "oracle 没有评估整个空间",
                                "evaluated": cell["n_evaluated_unique"],
                                "space": n})
        rows[op] = n
    return (not bad, f"预算与 oracle 全覆盖都成立：{not bad}；违反 {bad[:4]}",
            {"space_size": rows, "violations": bad[:4]})


# ===================================================================== 成本模型
@_case("baseline_speedup_is_exactly_one")
def _i_base_speedup(p):
    """基线的加速比恰好 1.0 —— 它就是分母。

    红的情况：基线的成本被另一条代码路径重算过（例如 `cost_sweep` 里
    重建了一个基线）—— 于是「加速比 8.5 倍」这个数的分母与基线表里
    印的那个数不是同一个，而两者都印在报告里。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        m = p["baseline.json"]["per_op"][op]["metrics"]
        rows[op] = {"latency_us": round(m["latency_us"], 9),
                    "bottleneck": m["bottleneck"]}
        if not (m["latency_us"] > 0):
            bad.append({"op": op, "why": "基线延迟非正"})
    # τ=0 的可行集里应当有一个「与基线逐位相同」的 kernel（漂移 0）
    for op in _ops(p):
        c = _cell(p, op, "reject", 0.0)["oracle"]
        b = c["best_gate1_only"]
        if b is None or b["drift"] != 0.0:
            bad.append({"op": op, "why": "τ=0 下没有漂移为 0 的可行 kernel"})
    return (not bad, f"基线是自洽的分母：{not bad}；违反 {bad}", rows)


@_case("speedup_and_throughput_are_collinear")
def _i_collinear(p):
    """`corr(speedup, throughput) == 1.0`（逐算子，容差 1e-6）。

    对固定算子，`throughput ≡ speedup × (flops / 基线延迟)` ——
    **两个目标轴在数学上是同一件事**。这不是 bug，是 README 那三条读数里的
    一个真实冗余；**必须报出来**，否则前沿看起来是三维的。

    红的情况：有人在成本模型里让吞吐不按延迟算（例如吞吐用了不同的 flops）——
    那样这条冗余消失了，而「三轴其实是两轴」这条结论就失去了证据。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        c = _space(p, op)["objective_correlation"].get("speedup", {}).get("throughput")
        rows[op] = c
        if c is None or c != c or abs(c - 1.0) > 1e-6:
            bad.append({"op": op, "corr": c})
    return (not bad, f"corr(speedup, throughput) = {rows}；违反 {bad}", rows)


@_case("cost_sweep_reproduces_the_default_launch_overhead_row")
def _i_cost_sweep(p):
    """成本扫描里 `launch_overhead_us = 默认值` 那一行的 speedup，
    必须等于交出去的答案的 speedup。

    红的情况：`cost_sweep` 用了另一个基因组（例如 `best_score` 而不是 `best`）——
    于是报告里「把 launch 开销扫到 0 后加速比缩小多少」这条结论，
    是关于**另一个 kernel** 的，而两个数印在同一节里。
    """
    cfg = _cfg(p)
    lo = float(cfg["cost"]["launch_overhead_us"])
    bad = []
    rows = {}
    for op in _ops(p):
        cs = p["benchmark.json"]["cost_sweep"].get(op)
        b = p["ladder.json"]["best_answer"].get(op)
        if not cs or b is None:
            bad.append({"op": op, "why": "缺成本扫描或头条答案"})
            continue
        row = next((r for r in cs["rows"] if abs(r["launch_overhead_us"] - lo) < 1e-12),
                   None)
        if row is None:
            bad.append({"op": op, "why": "扫描表里没有默认 launch 开销那一行"})
            continue
        rows[op] = {"sweep_speedup": row["speedup"],
                    "answer_speedup": round(b["objectives"][0], 9)}
        if abs(row["speedup"] - b["objectives"][0]) > 1e-6 * max(
                1.0, abs(b["objectives"][0])):
            bad.append({"op": op, "why": "扫描那一行与头条答案的 speedup 不同",
                        "sweep": row["speedup"],
                        "answer": b["objectives"][0]})
    return (not bad, f"成本扫描与头条答案口径一致：{not bad}；违反 {bad}", rows)


# ===================================================================== 基因表
@_case("gene_effects_partition_the_gene_set")
def _i_gene_partition(p):
    """`inert ∪ active ∪ unmeasured == 全部基因`，且两两不交。

    红的情况：某个基因掉进了两个桶（例如 `measured` 与 `inert` 的判据重叠）——
    于是「9 个基因里有 3 个惰性」这句话的分母是错的。
    """
    bad = []
    rows = {}
    for op in _ops(p):
        g = _genes(p, op)
        inert = set(g["numerically_inert_genes"])
        active = set(g["numerically_active_genes"])
        un = set(g["unmeasured_genes"])
        rows[op] = {"inert": sorted(inert), "active": sorted(active),
                    "unmeasured": sorted(un)}
        union = inert | active | un
        if union != set(GENE_ORDER):
            bad.append({"op": op, "why": "三个桶的并集不是全部基因",
                        "missing": sorted(set(GENE_ORDER) - union),
                        "extra": sorted(union - set(GENE_ORDER))})
        if inert & active or inert & un or active & un:
            bad.append({"op": op, "why": "三个桶有重叠",
                        "overlaps": sorted((inert & active) | (inert & un)
                                           | (active & un))})
        if len(g["rows"]) != len(GENE_ORDER):
            bad.append({"op": op, "why": "行数不等于基因数",
                        "rows": len(g["rows"])})
    return (not bad, f"三个桶是基因集的一个划分：{not bad}；违反 {bad}", rows)


@_case("unmeasured_gene_rows_carry_a_note")
def _i_unmeasured_note(p):
    """「没测到」的行必须显式标注 —— 它**不是**「没影响」。

    红的情况：`n_pairs == 0` 的行 `max_delta` 也是 0，读起来与
    「这个基因完全没用」一模一样。**这两件事必须能分开。**
    """
    bad = []
    for op in _ops(p):
        for r in _genes(p, op)["rows"]:
            if r["n_pairs"] == 0 and not str(r.get("note", "")).strip():
                bad.append({"op": op, "gene": r["gene"]})
            if r["n_pairs"] == 0 and r["measured"]:
                bad.append({"op": op, "gene": r["gene"], "why": "measured 与 n_pairs 矛盾"})
    return (not bad, f"「没测到」全部被显式标注：{not bad}；违反 {bad}", bad)


# ===================================================================== 产物与摘要
@_case("summary_is_a_projection_of_the_products")
def _i_summary(p):
    """`summary.json` 里的每个关键数字都必须能在别的产物里找到**同一个值**。

    红的情况：摘要是**另算一遍**的（而不是从产物里取）—— 于是它与产物
    可能不一致，而两者都印在报告里。摘要进了版本号，所以这条特别要紧：
    摘要一旦独立计算，「两个版本号相同 ⇒ 连验收结论都相同」就不再成立。
    """
    s = p["summary.json"]
    checks: list[tuple[str, Any, Any]] = [
        ("protocol_digest", s["protocol_digest"], p["config.json"]["protocol_digest"]),
        ("space_size_total", s["space_size_total"],
         sum(_space(p, op)["n_genomes"] for op in _ops(p))),
        ("n_best_answers", s["n_best_answers"],
         len(p["ladder.json"]["best_answer"])),
        ("acceptance.n", s["acceptance"]["n"], len(p["acceptance.json"]["checks"])),
        ("acceptance.n_passed", s["acceptance"]["n_passed"],
         p["acceptance.json"]["n_passed"]),
        ("expectations.n_held", s["expectations"]["n_held"],
         p["expectations.json"]["n_held"]),
        ("n_gate1_only_anywhere", s["n_gate1_only_anywhere"],
         sum(len(_adv(p, op)["gate1_only_anywhere"]) for op in _ops(p))),
    ]
    for op in _ops(p):
        checks.append((f"hypervolume_at_tau0[{op}]", s["hypervolume_at_tau0"][op],
                       p["ladder.json"]["frontier"][op]["hypervolume"]))
        checks.append((f"gate1_only_anywhere[{op}]", s["gate1_only_anywhere"][op],
                       _adv(p, op)["gate1_only_anywhere"]))
        checks.append((f"numerically_inert_genes[{op}]",
                       s["numerically_inert_genes"][op],
                       _genes(p, op)["numerically_inert_genes"]))
        checks.append((f"safe_window[{op}]", s["safe_window"][op],
                       _adv(p, op)["safe_window"]))
    bad = [{"key": k, "summary": a, "product": b}
           for k, a, b in checks if a != b]
    return (not bad,
            f"{len(checks)} 个摘要读数全部能在产物里找到同值：{not bad}；"
            f"不一致 {bad[:3]}",
            {"n_checked": len(checks), "violations": bad[:3]})


@_case("acceptance_counts_match_the_checks")
def _i_acc_counts(p):
    """验收表的计数与逐条结论一致，且**没有「没有检查函数认领」的门槛**。

    红的情况：某条门槛的名字拼错了 ⇒ 它被记成「没有任何检查函数认领」并
    **永远为红**，而真因（拼写）完全看不出来。这里把那种条目单独抓出来。
    """
    a = p["acceptance.json"]
    n_ok = sum(1 for c in a["checks"] if c["passed"])
    bad = []
    if a["n"] != len(a["checks"]):
        bad.append({"why": "n ≠ len(checks)", "n": a["n"], "len": len(a["checks"])})
    if a["n_passed"] != n_ok:
        bad.append({"why": "n_passed 与逐条结论不一致", "n_passed": a["n_passed"],
                    "recount": n_ok})
    if a["n_passed"] + a["n_failed"] != a["n"]:
        bad.append({"why": "通过 + 未通过 ≠ 总数"})
    unclaimed = [c["name"] for c in a["checks"]
                 if "没有任何检查函数认领" in str(c["detail"])]
    if unclaimed:
        bad.append({"why": "有门槛没有检查函数", "names": unclaimed})
    if a["failed"] != [c["name"] for c in a["checks"] if not c["passed"]]:
        bad.append({"why": "failed 列表与逐条结论不一致"})
    return (not bad, f"{a['n']} 条门槛的计数自洽：{not bad}；违反 {bad}", bad)


@_case("every_check_has_a_threshold_and_a_detail")
def _i_check_fields(p):
    """每条门槛都要有非空的 `threshold` 与 `detail`。

    红的情况：一条门槛的 detail 被写成空串 —— 于是报告里那一行只有一个
    「通过」，**读者无从知道它量了什么**，而它与其他行长得一样。
    """
    bad = []
    for c in p["acceptance.json"]["checks"]:
        if c.get("threshold") is None or c.get("threshold") == "":
            bad.append({"name": c["name"], "why": "threshold 为空"})
        if not str(c.get("detail", "")).strip():
            bad.append({"name": c["name"], "why": "detail 为空"})
    return (not bad, f"{p['acceptance.json']['n']} 条门槛都有阈值与说明：{not bad}",
            bad)


@_case("expectation_counts_match_the_items")
def _i_exp_counts(p):
    """期望的计数与逐条裁定一致，且**没有缺求值函数的期望**。

    红的情况：配置里声明了一条期望但没写求值函数 ⇒ 它被记成 `unmeasurable`
    并附一句「没有任何求值函数认领」—— 它看起来像一条设计承诺，
    其实没有人去核过。这里把它单独抓出来。
    """
    e = p["expectations.json"]
    bad = []
    n_held = sum(1 for i in e["items"] if i["verdict"] == "held")
    n_ref = sum(1 for i in e["items"] if i["verdict"] == "refuted")
    n_un = sum(1 for i in e["items"] if i["verdict"] == "unmeasurable")
    if (n_held, n_ref, n_un) != (e["n_held"], e["n_refuted"], e["n_unmeasurable"]):
        bad.append({"why": "计数与逐条裁定不一致",
                    "recount": [n_held, n_ref, n_un],
                    "recorded": [e["n_held"], e["n_refuted"], e["n_unmeasurable"]]})
    if e["n"] != len(e["items"]):
        bad.append({"why": "n ≠ len(items)"})
    orphan = [i["name"] for i in e["items"]
              if "没有任何求值函数认领" in str(i["detail"])]
    if orphan:
        bad.append({"why": "有期望没有求值函数", "names": orphan})
    if sorted(e["refuted_names"]) != sorted(
            i["name"] for i in e["items"] if i["verdict"] == "refuted"):
        bad.append({"why": "refuted_names 与逐条裁定不一致"})
    return (not bad, f"{e['n']} 条期望的计数自洽：{not bad}；违反 {bad}", bad)


@_case("timing_expectations_are_not_in_the_versioned_products")
def _i_timing(p):
    """时序类期望**必须**在 `benchmark.json` 里，**不许**出现在
    进版本摘要的 `expectations.json` 里。

    红的情况：它的证据里嵌着每次求值的微秒数，而 `expectations.json` 进摘要 ⇒
    「同一份输入跑两次」会得到两个版本号，**版本号的意义当场失效**。
    （`L6-graphopt` 实测抓过这个，附录 C 286。）
    """
    e = p["expectations.json"]
    b = p["benchmark.json"].get("timing_expectations", [])
    names_e = {i["name"] for i in e["items"]}
    names_b = {i["name"] for i in b}
    bad = []
    if "analytic_latency_tracks_wallclock" in names_e:
        bad.append({"why": "时序期望还在进摘要的产物里"})
    if "analytic_latency_tracks_wallclock" not in names_b:
        bad.append({"why": "时序期望没有落进 benchmark.json"})
    if names_e & names_b:
        bad.append({"why": "同一条期望同时出现在两处", "names": sorted(names_e & names_b)})
    if e["n_timing"] != len(b):
        bad.append({"why": "n_timing 与 benchmark 里的条数不一致",
                    "n_timing": e["n_timing"], "in_benchmark": len(b)})
    if "timing_items" in e:
        bad.append({"why": "timing_items 被整段复制进了 expectations.json"})
    return (not bad,
            f"时序读数只在 benchmark.json（{len(b)} 条）：{not bad}；违反 {bad}",
            {"in_expectations": sorted(names_e), "in_benchmark": sorted(names_b)})


@_case("versioned_files_are_all_present")
def _i_versioned(p):
    """`version.json` 里列出的每个进摘要的产物都必须真的在 `products` 里。

    红的情况：往 `VERSIONED_FILES` 加了一个名字但流水线不产它 ⇒
    `content_version` 会 `KeyError`（响亮），或者更糟 ——
    如果加了 `products.get(name, {})` 这样的兜底，版本号会**静默地**
    对一份空字典取摘要，于是它不再随那个产物变化。
    """
    listed = list(p["version.json"]["versioned_files"])
    missing = [n for n in listed if n not in p]
    # 反向：不在摘要里的名字不许出现在 VERSIONED_FILES 里
    extra = [n for n in listed if n not in VERSIONED_FILES and n != "version.json"]
    bad = []
    if missing:
        bad.append({"why": "摘要里列的产物缺失", "names": missing})
    if extra:
        bad.append({"why": "摘要里列了不该进摘要的产物", "names": extra})
    return (not bad,
            f"{len(listed)} 个进摘要的产物全部在位：{not bad}；违反 {bad}",
            {"versioned": listed, "missing": missing, "extra": extra})


@_case("products_are_json_serializable")
def _i_jsonable(p):
    """整个 `products` 必须能 JSON 序列化 —— **这是「渲染器只吃产物字典」
    这条纪律的前提**。

    红的情况：某个产物里塞了一个 `Trial` / `KernelGenome` / `set` ——
    那样报告在内存里渲染时一切正常（渲染器读得到那个对象），
    而 `visualize` 从 `out/` 重建时会**炸**。**两条路径不一致，
    而只有第二条会红。**

    ⚠️ **这里刻意不报「序列化后多少字节」。** 第一版报了，于是
    `acceptance.json`（进版本摘要）里嵌了一个**随时长变化**的数 ——
    两次 run 得到两个版本号，而这条用例本身**通过了**。
    **它抓到的是它自己**：一条用例的输出也是产物的一部分，
    所以它同样受「不许把时序读数写进进摘要的产物」这条纪律约束。
    """
    try:
        json.dumps({k: v for k, v in p.items()},
                   sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return False, f"**序列化失败**：{type(exc).__name__}: {exc}", None
    return (True, f"全部 {len(p)} 个产物可序列化", {"n_products": len(p)})


@_case("no_elapsed_time_key_in_the_versioned_products")
def _i_timing_keys(p):
    """进摘要的产物里**不许**出现「耗时读数」—— 它们只有一个出口：
    不进摘要的 `benchmark.json`。

    红的情况：把 `timings` 顺手挂到某个进摘要的产物上（或让一条用例把
    耗时写进自己的 `observed`）—— 于是「同一份输入跑两次得到同一个版本号」
    当场失效，而产物看起来只是「多了一个有用的字段」。
    **本项目实测被这条抓到过一次：一条用例把「序列化后多少字节」写进了
    `observed`，而那个数随 `benchmark.json` 里的耗时变。**

    ⚠️ 这条用例的**判据边界必须写清楚，否则它自己会变成假阳性制造机**
    （实测第一次上线就误报 3 处）。它判的是「**名字像耗时的东西底下的数字**」：
      ① 键名命中 `timings?/elapsed/duration/wall_clock/perf_counter`，
         且不是**基数词**（`n_*` / `*_count` / `num_*`）；
      ② 值是数字（`float`/`int`，`bool` 不算）——
         **或者它是那个键底下任意深度的数字叶子**
         （`timings: {"evaluate": 0.12}` 这种「字典包一层」是最常见的形态，
          第一版只认「键名像耗时**且值就是数字**」，于是这种形态整类漏掉）。
    于是三类东西被正确地放过去：
      - `expectations.json.n_timing = 1` —— 这是**计数**（有几个期望依赖耗时），
        不是读数，它本身是确定的；
      - `config.json.resolved.expectations.<id>` —— 这是**声明**（期望的 id 与
        claim），整段被显式跳过：那里的键名是 id，不是读数；
      - 任何 `expect: True` 之类的布尔。
    它抓不到的那一类 —— 「把耗时转成字符串嵌进一句文案里」—— 由 `verify`
    的两次 `run` 端到端兜底（字符串里的数字会让版本号动）。**这条用例的价值
    是在 `run` 当场给出信号，不是在替代 `verify`。**
    """
    pat = re.compile(r"(timings?|elapsed|duration|wall_?clock|perf_counter)",
                     re.IGNORECASE)
    cardinality = re.compile(r"^(n_|num_)|_count$", re.IGNORECASE)
    # 声明注册表：键名是**期望的 id**，不是读数 —— 整段跳过。
    decl_prefixes = (("config.json", "resolved", "expectations"),
                     ("config.json", "expectations"))
    hits: list[str] = []

    def walk(node: Any, path: tuple, under_timing: bool) -> None:
        if any(path[:len(pre)] == pre for pre in decl_prefixes):
            return
        if isinstance(node, dict):
            for k in sorted(node):
                key = str(k)
                val = node[k]
                hit_key = bool(pat.search(key)) and not cardinality.search(key)
                if (hit_key and isinstance(val, (int, float))
                        and not isinstance(val, bool)):
                    hits.append(f"{'.'.join(path)}.{key}={val!r}")
                walk(val, path + (key,), under_timing or hit_key)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if (under_timing and isinstance(v, (int, float))
                        and not isinstance(v, bool)):
                    hits.append(f"{'.'.join(path)}[{i}]={v!r}")
                walk(v, path + (f"[{i}]",), under_timing)
        elif (under_timing and isinstance(node, (int, float))
              and not isinstance(node, bool)):
            hits.append(f"{'.'.join(path)}={node!r}")

    for name in VERSIONED_FILES:
        if name in p:
            walk(p[name], (name,), False)
    return (not hits, f"进摘要的产物里的耗时读数：{hits or '（无）'}", hits)


@_case("no_timestamp_leaks_into_versioned_products")
def _i_timestamp(p):
    """进摘要的产物里**不许**出现 ISO 时间戳 —— 它只有一个出口：`version.json`。

    红的情况：有人把 `created_at` 顺手加进了 summary 或报告数据 ——
    于是「同一份输入跑两次得到同一个版本号」当场失效，
    而产物看起来只是「多了一个有用的字段」。
    """
    pat = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
    hits = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if pat.search(node):
                hits.append({"path": path, "value": node[:80]})
        elif isinstance(node, dict):
            for k in sorted(node):
                walk(node[k], f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node[:50]):
                walk(v, f"{path}[{i}]")

    for name in VERSIONED_FILES:
        if name in p:
            walk(p[name], name)
    return (not hits, f"进摘要的产物里的时间戳：{hits or '（无）'}", hits)


@_case("renderers_are_pure_functions_of_the_products")
def _i_render_pure(p):
    """把 `products` 过一遍 JSON 往返（丢掉一切对象身份）后重新渲染，
    两份报告必须**逐位相同**。

    红的情况：`render_html` 读了运行期对象（例如 `run.products` 的长度、
    或某个 `Trial`）—— 那样第一次渲染也许还能跑，
    而 `visualize` 从 `out/` 重建时会立刻炸。**这条用例让它在 run 里就红。**

    ⚠️ `version.json` 由 `emit.py` 写入；在 `finish` 阶段还没有它，
    所以流水线会先放一个**占位清单**（见 `pipeline.finish`）。
    这条用例同时核那个占位存在 —— 否则渲染器会在另一条路径上炸。
    """
    from .report import render_html, render_markdown
    if "version.json" not in p:
        return False, ("`products` 里没有 `version.json` —— 渲染器需要它。"
                       "（`pipeline.finish` 应当先放一个占位清单）"), None
    try:
        md1 = render_markdown(p)
        html1 = render_html(p)
    except Exception as exc:                        # noqa: BLE001
        return False, (f"**渲染器自己抛异常**：{type(exc).__name__}: {exc} "
                       "—— 说明它读了产物字典之外的东西"), None
    clone = json.loads(json.dumps(p, sort_keys=True, ensure_ascii=False))
    md2 = render_markdown(clone)
    html2 = render_html(clone)
    v = str(p["version.json"]["version"])
    bad = []
    if md1 != md2:
        bad.append("report.md 在 JSON 往返后不同")
    if html1 != html2:
        bad.append("report.html 在 JSON 往返后不同")
    if not md1.strip() or not html1.strip():
        bad.append("报告为空")
    if v not in md1 or v not in html1:
        bad.append(f"报告里没有版本号 {v!r}")
    # ⚠️ 这里**不报字节数** —— 报告正文里嵌了实测耗时表，字节数会随时长变，
    #    而这条用例的 `observed` 会被写进进摘要的 `acceptance.json`。
    #    （同一条纪律：用例的输出也是产物。）
    return (not bad,
            f"渲染是产物字典的纯函数：{not bad}；违反 {bad}",
            {"violations": bad})


# --------------------------------------------------------------------- 主入口
def run_cases(products: dict[str, Any]) -> dict[str, Any]:
    """跑完全部对抗用例。**用例自己抛异常也算失败**（不许吞）。"""
    cases: list[dict[str, Any]] = []
    for name in sorted(_CASES):
        try:
            ok, detail, observed = _CASES[name](products)
        except Exception as exc:                       # noqa: BLE001
            ok, detail, observed = (
                False,
                f"**用例自己抛异常**：{type(exc).__name__}: {exc} "
                f"—— 一个抛异常的用例与一条红的用例，都说明这里有东西不对",
                None)
        cases.append({"name": name, "ok": bool(ok), "detail": detail,
                      "observed": observed})
    failed = [c["name"] for c in cases if not c["ok"]]
    return {
        "cases": cases,
        "n_cases": len(cases),
        "n_cases_passed": sum(1 for c in cases if c["ok"]),
        "n_cases_failed": len(failed),
        "failed_cases": failed,
        "note": ("对抗用例是**不可调**的不变量断言：配置里没有它们的阈值。"
                 "红了说明代码或产物自相矛盾，不是「这次没达标」。"
                 "⚠️ `adversarial.json` 里那些**非法变体**是被测量的"
                 "实验对象（数据），不是对抗用例。"),
    }


def registered_cases() -> tuple[str, ...]:
    """用例名（供接线检查与门槛使用 —— 不跑用例就能知道有多少条）。"""
    return tuple(sorted(_CASES))


def cases_with_claims() -> dict[str, str]:
    """用例名 → 它的文档字符串（第一条 claim 行）。供报告与自检使用。"""
    out: dict[str, str] = {}
    for name, fn in _CASES.items():
        doc = (fn.__doc__ or "").strip().splitlines()
        out[name] = doc[0].strip() if doc else ""
    return out


__all__ = ["run_cases", "registered_cases", "cases_with_claims", "_CASES"]
