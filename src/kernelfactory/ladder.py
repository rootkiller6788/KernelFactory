"""阶梯：逐级放松**一条**约束（一致性容差 τ），看买到什么。

## 一条纪律：参考点必须**全空间唯一**

超体积的参考点只由空间决定（`min(各维) − pad`），与 τ / 策略 / 搜索器无关。
按级取参考点（`ref = min(该级可行域) − pad`）会让「放宽容差」同时下移参考点，
超体积凭空上涨 —— `L6-graphopt` 实测涨了 68%，**而那 68% 全是假象**
（附录 C 283）。这里沿用同一条保险丝 `reference_is_shared_across_rungs`。

## 两条「阶梯动没动」的读数

- `feasible_steps`：可行域**严格扩张**的级数；
- `active_steps`：**超体积严格上升**的级数。

⚠️ 本项目与 `L6-graphopt` 的分水岭就在 `active_steps` 上：
那边质量那一列**全平**（放宽容差只买到更大的可行域，买不到更好的答案）；
这边它**必须动**，因为精度轴同时进了算力与访存 ——
**一个全平的 `active_steps` 在这里意味着精度轴没接进成本模型，是 bug 不是结论。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .evaluate import (
    Trial, apply_gate, assign_scores, space_scale,
)
from .genome import KernelGenome
from .objective import OBJECTIVES
from .pareto import (
    correlation_matrix, distinct_count, hypervolume_2d, non_dominated_indices,
)
from .search import SearchContext, SearchResult, make_context, run_searchers


# --------------------------------------------------------------------------- #
# 单元
# --------------------------------------------------------------------------- #

@dataclass
class CellResult:
    """一个 `(搜索器, τ, 策略)` 格子的结论。

    ## ⚠️ 为什么这里存的是**字典**，不是 `Trial` 对象

    第一版把 `Trial` 对象直接存进来，然后在整个阶梯跑完后统一 `to_dict()`
    序列化。**那是错的**：`Trial` 是**可变对象**，被上百个格子共用，
    而 `apply_gate` 会不断改写它的 `feasible` / `deliverable` ——
    于是「τ=0.01 那一级的交付集」在产物里带着 **τ=1e-5 的门结论**。
    后果是 `max_answer_infeasible_rate` 与 `frontier_is_a_subset_of_the_deliverable_set`
    两条都红了，而它们看起来只是「门槛太严」。

    这个 bug 的形态与 `_frontier_of` 里那个「`on_frontier` 必须每次清零」
    是同一类：**跨格子共享可变状态**。区别是那个会在一次运行里累积，
    这个会在一次运行里被**后写的值覆盖前面的** —— 两者都不报错。
    ⇒ 格子一旦建好就**快照**成纯数据。

    ## 四个「最好」必须分清

    - `best`：**交付物** —— 门 1 ∧ 门 2 都过的最短延迟。
    - `best_gate1_only`：只过门 1 的最短延迟。
      **它与 `best` 的差距就是「单门槛设计」的代价** —— 一个可以印出来的数。
    - `best_score`：按评分最高的那个（`penalty` 策略下它可能不可行）。
    - 三者的判据都是「**它**可行吗」，不是「探索期候选的不可行率」。
    """

    searcher: str
    tau: float
    strategy: str
    best: dict[str, Any] | None = None
    best_gate1_only: dict[str, Any] | None = None
    best_score: dict[str, Any] | None = None
    n_feasible: int = 0
    n_deliverable: int = 0
    n_frontier: int = 0
    n_frontier_distinct: int = 0
    hypervolume: float = 0.0
    frontier: list[dict[str, Any]] = field(default_factory=list)
    n_evaluated_unique: int = 0
    budget_used: int = 0

    @property
    def best_latency_us(self) -> float | None:
        return None if self.best is None else float(self.best["latency_us"])

    @property
    def best_speedup(self) -> float | None:
        return None if self.best is None else float(self.best["speedup"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "tau": self.tau,
            "strategy": self.strategy,
            "best": self.best,
            "best_gate1_only": self.best_gate1_only,
            "best_score": self.best_score,
            "n_feasible": self.n_feasible,
            "n_deliverable": self.n_deliverable,
            "n_frontier": self.n_frontier,
            "n_frontier_distinct": self.n_frontier_distinct,
            "hypervolume": round(self.hypervolume, 12),
            "n_evaluated_unique": self.n_evaluated_unique,
            "budget_used": self.budget_used,
            "frontier": self.frontier,
        }


@dataclass
class Rung:
    op: str
    tolerance: float
    strategy: str
    reference: list[float]
    n_feasible_space: int
    cells: dict[str, CellResult] = field(default_factory=dict)

    def oracle(self) -> CellResult:
        return self.cells["oracle"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "tolerance": self.tolerance,
            "strategy": self.strategy,
            "reference": [round(x, 12) for x in self.reference],
            "n_feasible_space": self.n_feasible_space,
            "cells": {k: self.cells[k].to_dict() for k in sorted(self.cells)},
        }


# --------------------------------------------------------------------------- #
# 参考点与标尺
# --------------------------------------------------------------------------- #

def space_reference(trials: Sequence[Trial], pad: float) -> list[float]:
    """**全空间唯一**的参考点：`(min speedup, min effective_bits) − pad`。

    只用二维投影（speedup, effective_bits）—— 第三轴 `throughput` 与第一轴
    **完全共线**（见 `objective.py`），把它算进去只是把同一个体积换个刻度。
    """
    if not trials:
        raise ValueError("空空间没有参考点")
    lo_s = min(t.speedup for t in trials)
    lo_b = min(t.effective_bits for t in trials)
    return [lo_s - pad, lo_b - pad]


def _frontier_of(trials: Sequence[Trial]) -> list[Trial]:
    """交付物的前沿：在**过两道门**的集合上取非支配集。

    ⚠️ 前沿必须建在 `deliverable` 上，不能建在 `feasible` 上 ——
    否则前沿里会出现「声称 fp32、实际 8 位」的 kernel，
    而它看起来只是一个「又快又准」的点。
    """
    # ⚠️ 前沿标记必须**每次清零**：`Trial` 是可变对象，被上百个格子共用，
    #    不清零的话「这个点在前沿上」会**累积**成「它在任何一次里上过前沿」——
    #    一个跨格子污染的标志，看起来只是一个略大的前沿。
    for t in trials:
        t.on_frontier = False
        t.rank = -1
        t.dominated_by = 0
    pool = [t for t in trials if t.deliverable]
    if not pool:
        return []
    vecs = [t.objectives for t in pool]
    idx = non_dominated_indices(vecs)
    out = [pool[i] for i in sorted(idx, key=lambda i: (-pool[i].speedup,
                                                       pool[i].genome.digest()))]
    for rank, t in enumerate(out):
        t.on_frontier = True
        t.rank = rank
    # `dominated_by` 只在前沿内部为 0；其余点记「被几个前沿成员支配」
    for t in pool:
        if not t.on_frontier:
            t.dominated_by = sum(1 for f in out if _dom(f, t))
    return out


def _dom(a: Trial, b: Trial) -> bool:
    from .pareto import dominates
    return dominates(a.objectives, b.objectives)


def _cell(searcher: str, tau: float, strategy: str, trials: Sequence[Trial],
          reference: Sequence[float], n_eval: int, budget: int) -> CellResult:
    """建格子。**所有 Trial 在这里就地快照成字典**（见 `CellResult` 的文档）。"""
    feas = [t for t in trials if t.feasible]
    deliv = [t for t in trials if t.deliverable]
    front = _frontier_of(trials)
    fvec = [t.objectives for t in front]
    hv = hypervolume_2d(
        [[t.speedup, t.effective_bits] for t in front], reference)
    return CellResult(
        searcher=searcher, tau=tau, strategy=strategy,
        best=(min(deliv, key=lambda t: (t.latency_us, t.genome.digest())).to_dict()
              if deliv else None),
        best_gate1_only=(min(feas, key=lambda t: (t.latency_us,
                                                  t.genome.digest())).to_dict()
                         if feas else None),
        best_score=(max(trials, key=lambda t: (t.score,
                                              t.genome.digest())).to_dict()
                    if trials else None),
        n_feasible=len(feas), n_deliverable=len(deliv),
        n_frontier=len(front), n_frontier_distinct=distinct_count(fvec),
        hypervolume=hv, frontier=[t.to_dict(compact=True) for t in front],
        n_evaluated_unique=n_eval, budget_used=budget,
    )


# --------------------------------------------------------------------------- #
# 阶梯
# --------------------------------------------------------------------------- #

def _score_fn(trials_by_digest: dict[str, Trial], strategy: str, tau: float,
              cfg: dict[str, Any], scale: dict[str, float]):
    """给搜索器的打分函数。**它只看得到分数**（看不到成本、真值、容差）。"""
    pw = float(cfg["constraints"]["penalty_weight"])
    ds = float(cfg["constraints"]["drift_scale"])

    def f(gen: KernelGenome) -> float:
        t = trials_by_digest.get(gen.digest())
        if t is None:
            raise KeyError(f"搜索器问了一个空间外的基因组：{gen.label()}")
        return t.score
    return f


def run_ladder(cfg: dict[str, Any], op: str,
               trials: Sequence[Trial]) -> tuple[list[Rung], dict[str, float]]:
    """跑完整阶梯。**所有格子共用同一份已缓存的 trial**，不重算数值。"""
    scale = space_scale(trials)
    reference = space_reference(trials, float(cfg["search"]["reference_pad"]))
    tolerances = [float(x) for x in cfg["ladder"]["tolerances"]]
    strategies = ("reject", "penalty")
    by_digest = {t.genome.digest(): t for t in trials}
    genomes = [t.genome for t in trials]

    rungs: list[Rung] = []
    for tau in tolerances:
        for strat in strategies:
            apply_gate(trials, tau)
            assign_scores(trials, scale, strat, tau,
                          float(cfg["constraints"]["penalty_weight"]),
                          float(cfg["constraints"]["drift_scale"]))
            r = Rung(op=op, tolerance=tau, strategy=strat, reference=reference,
                     n_feasible_space=sum(1 for t in trials if t.feasible))
            names = list(cfg["search"]["searchers"]) + ["oracle"]
            for name in names:
                if name == "oracle":
                    r.cells[name] = _cell(name, tau, strat, trials, reference,
                                          len(trials), len(trials))
                    continue
                ctx = make_context(cfg, op, genomes,
                                   _score_fn(by_digest, strat, tau, cfg, scale))
                res: SearchResult = run_searchers(cfg, ctx, [name])[name]
                sub = [by_digest[d] for d in res.evaluated]
                r.cells[name] = _cell(name, tau, strat, sub, reference,
                                      len(res.evaluated), res.visits.used)
            rungs.append(r)
    return rungs, scale


def ladder_monotonicity(rungs: Sequence[Rung], strategy: str) -> dict[str, Any]:
    """阶梯单调性：τ 放松 ⇒ 可行域不减 ⇒ 最优目标不减（数学必然）。

    红了就是实现错了 —— 不是「这次没达标」。
    """
    rs = sorted([r for r in rungs if r.strategy == strategy],
                key=lambda r: r.tolerance)
    n_feas = [r.n_feasible_space for r in rs]
    n_deliv = [r.oracle().n_deliverable for r in rs]
    hv = [r.oracle().hypervolume for r in rs]
    best_lat = [r.oracle().best_latency_us for r in rs]
    best_sp = [r.oracle().best_speedup for r in rs]

    feasible_steps = sum(1 for a, b in zip(n_deliv, n_deliv[1:]) if b > a)
    active_steps = sum(1 for a, b in zip(hv, hv[1:]) if b > a + 1e-15)
    mono_f = all(b >= a for a, b in zip(n_deliv, n_deliv[1:]))
    mono_h = all(b >= a - 1e-15 for a, b in zip(hv, hv[1:]))
    lat = [x for x in best_lat if x is not None]
    #: 相对容差：延迟在产物里被舍入到 9 位小数，绝对容差会让「没变」被读成「涨了」。
    mono_l = all(b <= a * (1.0 + 1e-9) + 1e-12 for a, b in zip(lat, lat[1:])) \
        if all(x is not None for x in best_lat) else False
    return {
        "op": rs[0].op if rs else "",
        "strategy": strategy,
        "tolerances": [r.tolerance for r in rs],
        "n_feasible_space": n_feas,
        "n_deliverable_oracle": n_deliv,
        "hypervolume_oracle": [round(x, 12) for x in hv],
        "best_latency_us_oracle": [None if x is None else round(x, 9)
                                   for x in best_lat],
        "best_speedup_oracle": [None if x is None else round(x, 9)
                                for x in best_sp],
        "feasible_steps": feasible_steps,
        "active_steps": active_steps,
        "monotone_feasible": mono_f,
        "monotone_hypervolume": mono_h,
        "monotone_latency": mono_l,
        "monotone": mono_f and mono_h and mono_l,
        "note": "可行域单调是数学必然；**延迟单调下降不是必然** —— "
                "它取决于最优 kernel 在每一级是否真的换了。",
    }


# --------------------------------------------------------------------------- #
# 预算扫描
# --------------------------------------------------------------------------- #

def run_budget_sweep(cfg: dict[str, Any], op: str, trials: Sequence[Trial]
                     ) -> dict[str, Any]:
    """单一预算点不是比较，是轶事（附录 C 293）。

    在同一 τ / 策略下扫多个预算，看「搜索器的相对排序」稳不稳。
    随机搜索在大预算下会饱和 —— 于是**在那里比策略等于比运气**。
    """
    strat = str(cfg["budget_sweep"]["strategy"])
    tau = float(cfg["budget_sweep"]["tolerance"])
    apply_gate(trials, tau)
    scale = space_scale(trials)
    assign_scores(trials, scale, strat, tau,
                  float(cfg["constraints"]["penalty_weight"]),
                  float(cfg["constraints"]["drift_scale"]))
    reference = space_reference(trials, float(cfg["search"]["reference_pad"]))
    by_digest = {t.genome.digest(): t for t in trials}
    genomes = [t.genome for t in trials]
    budgets = [int(x) for x in cfg["budget_sweep"]["budgets"]]
    names = list(cfg["search"]["searchers"])
    oracle_hv = hypervolume_2d(
        [[t.speedup, t.effective_bits] for t in _frontier_of(trials)], reference)

    rows: list[dict[str, Any]] = []
    for b in budgets:
        row: dict[str, Any] = {"budget": b, "hypervolume": {}}
        for name in names:
            c2 = dict(cfg)
            c2["search"] = dict(cfg["search"])
            c2["search"]["budget_per_searcher"] = b
            ctx = make_context(c2, op, genomes, _score_fn(by_digest, strat, tau,
                                                          cfg, scale))
            res = run_searchers(c2, ctx, [name])[name]
            sub = [by_digest[d] for d in res.evaluated]
            hv = hypervolume_2d(
                [[t.speedup, t.effective_bits] for t in _frontier_of(sub)],
                reference)
            row["hypervolume"][name] = round(hv, 12)
        row["best_nonrandom"] = max(
            ((v, k) for k, v in row["hypervolume"].items() if k != "random"),
            default=(0.0, None))
        row["gain_over_random"] = round(
            row["best_nonrandom"][0] - row["hypervolume"]["random"], 12)
        rows.append(row)

    # 随机搜索在哪个预算上饱和（拿到 oracle 的全部超体积）
    sat = next((r["budget"] for r in rows
                if abs(r["hypervolume"]["random"] - oracle_hv) <= 1e-12), None)
    tight = int(cfg["search"]["tight_budget"])
    margins = [{"budget": r["budget"],
                "gain_over_random": r["gain_over_random"],
                "best_searcher": r["best_nonrandom"][1]} for r in rows]
    return {
        "op": op,
        "strategy": strat,
        "tolerance": tau,
        "budgets": budgets,
        "searchers": names,
        "oracle_hypervolume": round(oracle_hv, 12),
        "rows": rows,
        "margins": margins,
        "tight_budget": tight,
        "random_saturation_budget": sat,
        "note": "随机搜索在饱和点之后拿到 oracle 的全部超体积 —— "
                "在那里比策略等于比运气。",
    }


__all__ = ["CellResult", "Rung", "run_ladder", "ladder_monotonicity",
           "run_budget_sweep", "space_reference", "_frontier_of"]
