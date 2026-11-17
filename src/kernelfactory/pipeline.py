"""编排：把一次 `run` 的全部产物组装出来。

## 一条纪律：渲染器只吃产物字典

`build()` 返回的 `Run.products` 是**纯数据**（可 JSON 序列化）。
`report.render_markdown(products)` / `render_html(products)` **只读这个字典**，
不读 `KernelEvaluator` / `NumericsCache` / `Trial` 这些运行期对象。

买到两样：

1. `visualize` 能从 `out/` 重建报告，**一行都不重跑流水线**；
2. 它证明的东西更强了 —— 从「流水线是确定的」变成「**产物足以重造报告**」。
   对抗用例：把一次 run 折成产物 → 从产物重建 → 再渲染，与直接渲染**逐位比**。

## 什么进版本摘要，什么不进

`VERSIONED_FILES` 显式列出来。排除三类：

- `version.json`：自指 + 时间戳；
- `benchmark.json`：**耗时**（随机器变，算进去版本号就失去意义）；
- `report.md` / `report.html`：它们是**视图**。

⇒ 「两次 `run` 只有 `version.json` 不同」这条核对要**精确表述**为
「两次 `run` 只有 `version.json`（时间戳）与 `benchmark.json`（耗时）不同，
其余 N 个结论性产物逐位相同」。

## 三个算子必须**各自独立**跑完整条链

不是「一个算子跑完再复用」。每条链都是：空间 → 评估 → 阶梯 → 基因 → 对抗。
**共用的只有配置与成本常数** —— 因为「tile_k 值不值钱」这个问题在 matmul
与 layernorm 上的答案本来就不同（一个是算力瓶颈，一个是访存瓶颈）。
把三个算子混成一个池子，会让两个不同的答案互相抵消成一个没有意义的平均数。

## `summary.json` 必须排在对抗用例之前算

用例要读摘要（「摘要是不是别的产物的投影」本身就是一条用例），
而摘要要读验收结论 ⇒ 顺序只能是「门槛 → 摘要 → 用例」。
反过来的话那条用例只能跳过，而「跳过」与「通过」在输出里长得一模一样。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .adversarial import cross_op_headline, run_adversarial
from .benchmark import wallclock_probe
from .config import protocol_digest
from .cost import COST_SOURCE, cost_sweep
from .evaluate import (
    KernelEvaluator, Trial, apply_gate, space_scale,
)
from .genes import gene_coverage, gene_effects
from .genome import ROW_CONSTANT_GENES, enumerate_space, space_digest
from .ladder import (
    _frontier_of, ladder_monotonicity, run_budget_sweep, run_ladder,
    space_reference,
)
from .objective import OBJECTIVES, as_dict as objective_dict
from .pareto import correlation_matrix, distinct_count
from .search import declared_params

#: 进版本摘要的产物（**顺序固定**，摘要按这个顺序拼）。
VERSIONED_FILES: tuple[str, ...] = (
    "config.json",
    "baseline.json",
    "space.json",
    "genes.json",
    "adversarial.json",
    "search.json",
    "budget_sweep.json",
    "ladder.json",
    "expectations.json",
    "acceptance.json",
    "summary.json",
)

#: 落盘但不进摘要（耗时 / 视图 / 自指）。
UNVERSIONED_FILES: tuple[str, ...] = (
    "version.json", "benchmark.json", "report.md", "report.html",
)


@dataclass
class Run:
    cfg: dict[str, Any]
    products: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    #: 运行期对象，**渲染器不许读它们**（只用来给 `check` 打印诊断）。
    evaluators: dict[str, KernelEvaluator] = field(default_factory=dict)
    space_diag: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 逐算子的产物
# --------------------------------------------------------------------------- #

def _space_product(cfg: dict[str, Any], op: str, trials: Sequence[Trial],
                   diag: dict[str, Any], evaluator: KernelEvaluator,
                   rung0, reference: Sequence[float],
                   default_tol: float) -> dict[str, Any]:
    """搜索空间本身的刻画：多大、有多少不同的读数、目标之间冗不冗余。"""
    objs = [t.objectives for t in trials]
    corr = correlation_matrix(objs, OBJECTIVES) if len(trials) >= 3 else {}

    # 默认容差下，三个精度档各有多少可行 —— 用来核「默认容差收不收得下 bf16」
    apply_gate(trials, default_tol)
    feas_by_dtype: dict[str, int] = {}
    for t in trials:
        if t.feasible:
            k = t.genome.accum_dtype
            feas_by_dtype[k] = feas_by_dtype.get(k, 0) + 1

    n_legal_rejected = sum(1 for t in trials if t.feasible and not t.bits_ok)

    # ⚠️ `n_structural_failures` **必须是真的算出来的**。
    #    第一版把它写成常量 0，于是门槛 `max_structural_failures` 变成一句恒真的话 ——
    #    **一条恒真的判据与没有判据在输出里长得一模一样。**
    #    这里量的是「数值/成本模型给出了不可用的数」：延迟、加速比、吞吐、误差、
    #    有效位宽里出现 nan / inf。它真的会发生（除零、溢出），而且发生时
    #    所有下游读数都会变成 nan，而 nan 在表格里只是一个「—」。
    _NUMERIC_FIELDS = ("latency_us", "speedup", "throughput_tflops",
                       "drift", "error_vs_reference", "effective_bits")
    bad: list[dict[str, Any]] = []
    for t in trials:
        hit = [f for f in _NUMERIC_FIELDS
               if not math.isfinite(float(getattr(t, f)))]
        if hit:
            bad.append({"genome": t.genome.label(), "fields": hit})

    # 有效位宽的**实测**分布：与声明位宽分开报（它们不是一回事）
    by_bits: dict[str, list[float]] = {}
    for t in trials:
        by_bits.setdefault(t.genome.accum_dtype, []).append(t.effective_bits)

    # 抽样本给对抗用例核 √n 偏置（每个精度档取漂移最大/最小的两个）
    samples: list[Trial] = []
    for dt in sorted(by_bits):
        sub = sorted([t for t in trials if t.genome.accum_dtype == dt],
                     key=lambda t: (t.drift, t.genome.digest()))
        if sub:
            samples.append(sub[0])
            samples.append(sub[-1])

    return {
        "op": op,
        "row_wise": bool(cfg["ops"][op]["row_wise"]),
        "probe_shape": list(cfg["ops"][op]["probe"]),
        "production_shape": list(cfg["ops"][op]["production"]),
        "n_genomes": len(trials),
        "n_raw": diag["n_raw"],
        "n_valid_before_dedup": diag["n_valid"],
        "n_pruned": diag["n_pruned"],
        "n_collapsed_duplicates": diag["n_collapsed_duplicates"],
        "prune_reasons": diag["prune_reasons"],
        "gene_cardinality": diag["gene_cardinality"],
        "non_searching_genes": sorted(
            [n for n in ROW_CONSTANT_GENES
             if bool(cfg["ops"][op]["row_wise"])]),
        "space_digest": space_digest(trials_genomes(trials)),
        "n_distinct_genome_digests": len({t.genome.digest() for t in trials}),
        "n_distinct_objective_vectors": distinct_count(objs),
        "n_structural_failures": len(bad),
        "structural_failures": bad[:8],
        "structural_failure_note": "不可用的数（nan / inf）出现在延迟、加速比、"
                                   "吞吐、误差或有效位宽里。**它是算出来的，不是常量 0** ——"
                                   "一个恒真的判据与没有判据在输出里长得一样。",
        "accum_dtype_cardinality": sorted(
            {t.genome.accum_dtype for t in trials}),
        "n_distinct_accum_dtypes": len({t.genome.accum_dtype for t in trials}),
        "n_distinct_effective_bits": distinct_count(
            [(t.effective_bits,) for t in trials]),
        "effective_bits_by_dtype": {
            k: {"min": round(min(v), 6), "max": round(max(v), 6),
                "n": len(v)}
            for k, v in sorted(by_bits.items())},
        "declared_bits_by_dtype": {
            k: sorted({t.declared_bits for t in trials
                       if t.genome.accum_dtype == k})
            for k in sorted(by_bits)},
        "feasible_by_dtype_at_default_tolerance": feas_by_dtype,
        "default_tolerance": default_tol,
        "n_legal_rejected_by_gate2": n_legal_rejected,
        "objective_correlation": corr,
        "objective_scale": space_scale(trials),
        "n_frontier_at_tau0": rung0.oracle().n_frontier,
        "n_frontier_distinct_at_tau0": rung0.oracle().n_frontier_distinct,
        "hypervolume_at_tau0": round(rung0.oracle().hypervolume, 12),
        "reference": [round(x, 12) for x in reference],
        "reference_note": "全空间唯一参考点：min(整个空间) − reference_pad。"
                          "与容差 / 策略 / 搜索器全部无关。",
        "sample_trials": [t.to_dict() for t in samples],
        "evaluator": evaluator.diagnostics(),
    }


def trials_genomes(trials: Sequence[Trial]) -> list[Any]:
    return [t.genome for t in trials]


def _baseline_product(cfg: dict[str, Any], per_op: dict[str, Any]) -> dict[str, Any]:
    """基线：**每个算子一个手写 kernel**，它是所有 drift 的比较基准。"""
    rows: dict[str, Any] = {}
    for op, d in sorted(per_op.items()):
        bm = d["metrics"]
        rows[op] = {
            "genome": d["genome"].to_dict(),
            "label": d["genome"].label(),
            "genome_id": d["genome"].id,
            "row_wise": bool(cfg["ops"][op]["row_wise"]),
            "probe_shape": list(cfg["ops"][op]["probe"]),
            "production_shape": list(cfg["ops"][op]["production"]),
            "metrics": {k: bm[k] for k in sorted(bm)},
            "drift_is_zero_by_definition": True,
            "note": "基线自己的 drift **恰好是 0**（它就是比较基准）；"
                    "它的指标定义了 speedup 的分母。",
        }
    return {
        "per_op": rows,
        "cost_source": {k: COST_SOURCE[k] for k in sorted(COST_SOURCE)},
        "note": "成本模型里的**每一个常数都是假设**（见 cost_source 的 kind）；"
                "本机没有可用 GPU，也没有 torch / triton。",
    }


def _search_product(cfg: dict[str, Any], per_op: dict[str, Any],
                    cache: dict[str, Any]) -> dict[str, Any]:
    """搜索器清单与**预算不对称**（必须显式，见 `search.py` 的纪律 4）。"""
    space_sizes = {op: per_op[op]["n_genomes"] for op in sorted(per_op)}
    budget = int(cfg["search"]["budget_per_searcher"])
    return {
        "searchers": list(cfg["search"]["searchers"]) + ["oracle"],
        "params_declared": {k: list(v) for k, v in sorted(declared_params().items())},
        "space_size_per_op": space_sizes,
        "budget_per_searcher": budget,
        "budget_asymmetry": {
            "oracle": "= 空间大小（逐算子）",
            "others": budget,
            "note": "oracle 的预算 = 空间大小，其余 = budget_per_searcher。"
                    "这不是特权，是定义 —— 否则 oracle 就不是 oracle，"
                    "而 `oracle_is_optimal` 会从数学必然退化成「可能」。",
        },
        "cache": cache,
        "constraints": dict(cfg["constraints"]),
        "semantics": dict(cfg["semantics"]),
        "objectives": objective_dict(),
        "scoring": {
            "kind": "equal_weight_tchebycheff",
            "weights": [1.0 / len(OBJECTIVES)] * len(OBJECTIVES),
            "ideal": [1.0] * len(OBJECTIVES),
            "note": "固定权重只决定「搜索器看到的那个数」；"
                    "交付物是未加权的三目标前沿。"
                    "⚠️ 门 2 **不参与打分** —— 它是诚实性检查，不是优化目标。",
        },
    }


def _ladder_product(cfg: dict[str, Any], op: str, rungs: Sequence[Any],
                    reference: Sequence[float]) -> dict[str, Any]:
    """一个算子的阶梯：每一级、每种策略下 oracle 的前沿（基因组 + 口径）。"""
    out: dict[str, Any] = {}
    for r in rungs:
        key = f"{r.strategy}@{r.tolerance:g}"
        c = r.oracle()
        out[key] = {
            "op": op,
            "tolerance": r.tolerance,
            "strategy": r.strategy,
            "reference": [round(x, 12) for x in r.reference],
            "n_feasible_space": r.n_feasible_space,
            "cells": {k: r.cells[k].to_dict() for k in sorted(r.cells)},
        }
    return out


def _frontier_product(cfg: dict[str, Any], per_op: dict[str, Any]) -> dict[str, Any]:
    """交付物本体：τ=0 / reject 下每个算子的前沿（基因组 + 口径）。"""
    out: dict[str, Any] = {}
    for op, rungs in sorted(per_op.items()):
        key = "reject@0"
        if key not in rungs:
            continue
        cell = rungs[key]["cells"]["oracle"]
        out[op] = {
            "tolerance": rungs[key]["tolerance"],
            "strategy": rungs[key]["strategy"],
            "n_frontier": cell["n_frontier"],
            "n_frontier_distinct": cell["n_frontier_distinct"],
            "hypervolume": cell["hypervolume"],
            "points": [p for p in cell["frontier"]],
        }
    return out


def _best_answer_product(cfg: dict[str, Any],
                         per_op: dict[str, Any]) -> dict[str, Any]:
    """τ=0 / reject 下 oracle 交出去的那个答案（报告的头条数字来自这里）。"""
    out: dict[str, Any] = {}
    for op, rungs in sorted(per_op.items()):
        key = "reject@0"
        if key not in rungs:
            continue
        b = rungs[key]["cells"]["oracle"]["best"]
        if b is not None:
            out[op] = b
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def build(cfg: dict[str, Any]) -> Run:
    """跑完一次完整流水线，返回全部产物。**不写盘**（写盘在 `emit.py`）。"""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    ops = sorted(cfg["ops"])
    tolerances = [float(x) for x in cfg["ladder"]["tolerances"]]
    default_tol = float(cfg["semantics"]["tolerance"])

    evaluators: dict[str, KernelEvaluator] = {}
    space_diag: dict[str, Any] = {}
    per_space: dict[str, Any] = {}
    per_genes: dict[str, Any] = {}
    per_adv: dict[str, Any] = {}
    per_rungs: dict[str, Any] = {}
    per_sweep: dict[str, Any] = {}
    per_base: dict[str, Any] = {}
    monotonicity: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for op in ops:
        genomes, diag = enumerate_space(cfg, op)
        space_diag[op] = diag
        ev = KernelEvaluator(cfg, op)
        evaluators[op] = ev
        trials = ev.space(genomes)

        reference = space_reference(trials, float(cfg["search"]["reference_pad"]))
        rungs, _scale = run_ladder(cfg, op, trials)

        rung0 = next(r for r in rungs
                     if r.strategy == "reject" and r.tolerance == min(tolerances))
        per_space[op] = _space_product(cfg, op, trials, diag, ev, rung0,
                                       reference, default_tol)
        per_genes[op] = gene_effects(trials, op)
        per_genes[op]["coverage"] = gene_coverage(trials)
        per_rungs[op] = _ladder_product(cfg, op, rungs, reference)
        per_base[op] = {
            "genome": ev.baseline,
            "metrics": dict(ev.base_cost),
        }
        for s in ("reject", "penalty"):
            monotonicity.append(ladder_monotonicity(rungs, s))
        per_sweep[op] = run_budget_sweep(cfg, op, trials)

        # ⚠️ 对抗必须在 `apply_gate` 把 trials 恢复成默认容差之后跑 ——
        #    `safe_window` 读 `ev.space_all` 的漂移（与门无关），
        #    但 `by_tolerance` 的读数要靠 `run_adversarial` 自己重算，
        #    所以这里先把门重置回默认值，免得留下阶梯顶端的残留状态。
        apply_gate(trials, default_tol)
        adv = run_adversarial(cfg, ev, tolerances)
        adv["headline"] = cross_op_headline(adv, ev)
        per_adv[op] = adv

    timings["main"] = time.perf_counter() - t0

    # ---- 基准：解析延迟 vs 实测耗时（**只对前沿成员测**，省时间且信息量足）
    #
    # ⚠️ 前沿**从运行期的 Trial 对象重算**，不从产物里的紧凑前沿反序列化 ——
    #    `ladder.json` 里的前沿点是 `compact=True` 序列化的（没有 `genome`），
    #    拿它去 `Trial.from_dict` 会炸，而更糟的兜底是**静默造一个基因全为 0 的
    #    假 kernel** 然后照算 —— 那个结果看起来是个合法的数。
    t0 = time.perf_counter()
    bench: dict[str, Any] = {"wallclock": {}, "cost_sweep": {}}
    tau0 = min(tolerances)
    for op in ops:
        ev = evaluators[op]
        trials = ev.space_all
        apply_gate(trials, tau0)
        front = _frontier_of(trials)
        probe = front[:8]
        analytic = {t.genome.digest(): t.latency_us for t in probe}
        bench["wallclock"][op] = wallclock_probe(
            cfg, ev, [t.genome for t in probe], analytic)
        deliv = [t for t in trials if t.deliverable]
        if deliv:
            best = min(deliv, key=lambda t: (t.latency_us, t.genome.digest()))
            bench["cost_sweep"][op] = cost_sweep(
                cfg, op, [], {"genome": ev.baseline}, {"genome": best.genome})
    timings["benchmark"] = time.perf_counter() - t0

    # 缓存诊断：逐算子加起来（**键里带 op 名**，见 search.py 纪律 2）
    cache = {
        "per_op": {op: evaluators[op].num.diagnostics() for op in ops},
        "unique_evaluations": sum(evaluators[op].num.misses for op in ops),
        "cache_hits": sum(evaluators[op].num.hits for op in ops),
    }
    cache["hit_rate"] = (cache["cache_hits"]
                         / max(1, cache["cache_hits"] + cache["unique_evaluations"]))
    cache["note"] = ("内核输出只由「归约结构 + 精度」决定 —— 命中率高说明"
                     "**绝大部分基因对数值结果没有影响**，"
                     "而这正是「基因惰性」那张表的数值侧证据。")

    products: dict[str, Any] = {
        "config.json": {"resolved": cfg, "protocol_digest": protocol_digest(cfg)},
        "baseline.json": _baseline_product(cfg, per_base),
        "space.json": {
            "per_op": per_space,
            "n_ops": len(ops),
            "ops": ops,
            "objective_correlation_pooled": correlation_matrix(
                [o for op in ops for o in
                 [t["objectives"] for t in per_space[op]["sample_trials"]]],
                OBJECTIVES) if all(len(per_space[op]["sample_trials"]) >= 3
                                   for op in ops) else {},
            "pooled_note": "⚠️ 这是**把三个算子混在一起**算的相关矩阵，"
                           "只用来说明「算子之间差多少」。"
                           "「三个目标轴冗不冗余」这个问题必须**逐算子**回答 —— "
                           "见 `per_op[op].objective_correlation`。",
        },
        "genes.json": {"per_op": per_genes},
        "adversarial.json": {
            "per_op": per_adv,
            "note": "⚠️ 这里的**非法变体**是**被测量的实验对象**（数据），"
                    "**不是**对抗用例。对抗用例是 `acceptance.json.cases` 里"
                    "那些不可调的不变量断言。**同一个词指两个东西，"
                    "读者没法从报告内部发现。**",
        },
        "search.json": _search_product(cfg, per_space, cache),
        "budget_sweep.json": {"per_op": per_sweep},
        "ladder.json": {
            "per_op": per_rungs,
            "monotonicity": monotonicity,
            "frontier": _frontier_product(cfg, per_rungs),
            "best_answer": _best_answer_product(cfg, per_rungs),
        },
        "benchmark.json": bench,
        "summary.json": {},      # 下面填（依赖 expectations / acceptance）
    }

    bench["timings"] = dict(timings)
    timings["total"] = time.perf_counter() - t_start
    bench["timings"] = dict(timings)

    return Run(cfg=cfg, products=products, timings=timings,
               evaluators=evaluators, space_diag=space_diag)


def finish(run: Run) -> Run:
    """补上依赖 expectations / acceptance 的产物，并算 summary。

    **分两步是为了让 `summary` 能引用验收结果** —— 而 `summary` 又进版本摘要，
    所以「验收通过与否」会进版本号。这是刻意的：
    两个版本号相同 ⇒ 连验收结论都相同，而不只是数字相同。
    """
    from .acceptance import run_acceptance
    from .expectations import run_expectations
    from .invariants import registered_cases, run_cases

    products = run.products
    cfg = run.cfg
    ops = sorted(cfg["ops"])

    exp_all = run_expectations(products)
    # ⚠️ **时序类期望必须落进 benchmark.json**（不进版本摘要的产物）。
    #    `analytic_latency_tracks_wallclock` 的证据里嵌着每次求值的微秒数，
    #    而 `expectations.json` 是**进摘要**的 ⇒ 「同一份输入跑两次」
    #    会得到两个版本号，版本号的意义当场失效。
    products["benchmark.json"]["timing_expectations"] = exp_all["timing_items"]
    products["expectations.json"] = {k: v for k, v in exp_all.items()
                                     if k != "timing_items"}
    products["acceptance.json"] = run_acceptance(products)

    exp = products["expectations.json"]
    acc = products["acceptance.json"]
    ladder = products["ladder.json"]
    space = products["space.json"]
    adv = products["adversarial.json"]
    genes = products["genes.json"]
    sweep = products["budget_sweep.json"]
    base = products["baseline.json"]

    best = ladder["best_answer"]
    r0 = ladder["frontier"]

    # ---- 头条对照的**跨算子**汇总（报告那一节的数字来自这里）
    headlines = {op: adv["per_op"][op].get("headline", {}) for op in ops}
    ok_ops = [op for op in ops if headlines[op].get("available")]
    ratios = {op: headlines[op].get("drift_ratio") for op in ok_ops}
    gate1_agrees = {op: bool(headlines[op].get("gate1_same_verdict"))
                    for op in ok_ops}
    gate2_sep = {op: bool(headlines[op].get("gate2_separates"))
                 for op in ok_ops}
    inside = {op: bool(headlines[op].get("fake_inside_honest_range"))
              for op in ok_ops}
    bit_identical = {op: bool(
        headlines[op].get("nearest_honest_bf16", {}).get("bit_identical_drift"))
        for op in ok_ops}

    products["summary.json"] = {
        "protocol_digest": products["config.json"]["protocol_digest"],
        "n_ops": len(ops),
        "space_size_per_op": {op: space["per_op"][op]["n_genomes"] for op in ops},
        "space_size_total": sum(space["per_op"][op]["n_genomes"] for op in ops),
        "n_distinct_objective_vectors_per_op": {
            op: space["per_op"][op]["n_distinct_objective_vectors"]
            for op in ops},
        "baseline": {
            op: {
                "label": base["per_op"][op]["label"],
                "latency_us": round(base["per_op"][op]["metrics"]["latency_us"], 9),
                "bottleneck": base["per_op"][op]["metrics"]["bottleneck"],
                "efficiency": round(base["per_op"][op]["metrics"]["efficiency"], 9),
            }
            for op in ops},
        "best_answer": best,
        "n_best_answers": len(best),
        "frontier_points_at_tau0": {
            op: r0[op]["n_frontier"] for op in ops if op in r0},
        "frontier_points_distinct_at_tau0": {
            op: r0[op]["n_frontier_distinct"] for op in ops if op in r0},
        "hypervolume_at_tau0": {op: r0[op]["hypervolume"]
                                for op in ops if op in r0},
        "safe_window": {op: adv["per_op"][op]["safe_window"] for op in ops},
        "safe_window_nonempty": {op: adv["per_op"][op]["safe_window"]["nonempty"]
                                 for op in ops},
        "illegal_accepted_at_default_tolerance": {
            op: adv["per_op"][op]["n_accepted_at_default_tolerance"]
            for op in ops},
        "gate1_only_anywhere": {
            op: adv["per_op"][op]["gate1_only_anywhere"] for op in ops},
        "n_gate1_only_anywhere": sum(
            len(adv["per_op"][op]["gate1_only_anywhere"]) for op in ops),
        #: **本项目的头条**：诚实 bf16 与「假装 fp32」的漂移之比
        #: （取**最难区分的那一对**）。比值 ≈ 1 意味着门 1 对两者给出同一个结论。
        "headline_drift_ratio": ratios,
        "headline_gate1_agrees": gate1_agrees,
        "headline_gate2_separates": gate2_sep,
        "headline_fake_inside_honest_range": inside,
        "headline_bit_identical_drift": bit_identical,
        "ladder_best_speedup_oracle": {
            op: next(r["best_speedup_oracle"]
                     for r in ladder["monotonicity"]
                     if r["op"] == op and r["strategy"] == "reject")
            for op in ops},
        "ladder_active_steps": {
            r["strategy"]: r["active_steps"]
            for r in ladder["monotonicity"] if r["op"] == ops[0]},
        "ladder_active_steps_per_op": {
            op: next(r["active_steps"] for r in ladder["monotonicity"]
                     if r["op"] == op and r["strategy"] == "reject")
            for op in ops},
        "ladder_feasible_steps_per_op": {
            op: next(r["feasible_steps"] for r in ladder["monotonicity"]
                     if r["op"] == op and r["strategy"] == "reject")
            for op in ops},
        "numerically_inert_genes": {
            op: genes["per_op"][op]["numerically_inert_genes"] for op in ops},
        "numerically_active_genes": {
            op: genes["per_op"][op]["numerically_active_genes"] for op in ops},
        "unmeasured_genes": {
            op: genes["per_op"][op]["unmeasured_genes"] for op in ops},
        "budget_sweep": {
            "budgets": sweep["per_op"][ops[0]]["budgets"],
            "random_saturation_budget": {
                op: sweep["per_op"][op]["random_saturation_budget"]
                for op in ops},
            "tight_budget": int(cfg["search"]["tight_budget"]),
            "tight_budget_best_gain_over_random": {
                op: next((m["gain_over_random"] for m in
                          sweep["per_op"][op]["margins"]
                          if m["budget"] == int(cfg["search"]["tight_budget"])),
                         None)
                for op in ops},
        },
        "expectations": {
            "n": exp["n"],
            "n_held": exp["n_held"],
            "n_refuted": exp["n_refuted"],
            "n_unmeasurable": exp["n_unmeasurable"],
            #: 时序类期望的**条数**是确定性的（内容不是）——
            #: 它们的读数在 `benchmark.json` 里。
            "n_timing_dependent": exp["n_timing"],
            "refuted_names": exp["refuted_names"],
        },
        "acceptance": {
            "n": len(acc["checks"]),
            "n_passed": acc["n_passed"],
            "n_failed": acc["n_failed"],
            #: 对抗用例的条数也进摘要 —— 但它**不是**门槛，所以不进 `n`。
            #: 口径：门槛是达标线（可调），对抗用例是不变量断言（不可调）。
            #: ⚠️ 这里读**注册表**而不是 `acceptance.json`：摘要必须排在
            #:    用例之前算（用例要读摘要），此刻 `cases` 还没被填进去 ——
            #:    读产物会永远读到 0，而 0 看起来像个合法的数。
            "n_cases": len(registered_cases()),
            "n_cases_failed": None,   # 用例跑完后回填（见函数末尾）
        },
        "cache": {k: cache_v for k, cache_v in products["search.json"]["cache"].items()
                  if k != "per_op"},
    }

    # ---- 对抗用例：**必须放在 summary 之后**
    # 它们要读 `summary.json`（「摘要是不是别的产物的投影」本身就是一条用例），
    # 而 `summary.json` 又要读验收结论 ⇒ 顺序只能是「门槛 → 摘要 → 用例」。
    # 反过来的话那条用例只能跳过，而「跳过」与「通过」在输出里长得一模一样。
    #
    # ⚠️ 渲染器要读 `version.json`，而它由 `emit()` 写入 —— `finish` 阶段还没有。
    #    这里先放一个**占位清单**（`emit` 会覆盖它），于是
    #    「渲染器只吃产物字典」这条用例能在**同一条代码路径**上核到真实渲染。
    #    不放占位的话那条用例只能跳过，而「跳过」与「通过」长得一样。
    products.setdefault("version.json", {
        "version": "(占位：emit 尚未运行)",
        "versioned_files": list(VERSIONED_FILES),
        "unversioned_files": list(UNVERSIONED_FILES),
        "digests": {},
        "created_at": None,
        "note": "占位版本清单 —— `emit()` 会用真正的清单覆盖它。",
    })
    cases = run_cases(products)
    products["acceptance.json"].update(cases)
    # 回填摘要里那个只有跑完才知道的数（条数已知，失败数未知）。
    products["summary.json"]["acceptance"]["n_cases_failed"] = cases["n_cases_failed"]
    return run


__all__ = ["Run", "VERSIONED_FILES", "UNVERSIONED_FILES", "build", "finish"]
