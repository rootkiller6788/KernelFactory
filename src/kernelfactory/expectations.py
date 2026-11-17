"""设计期望：**允许被实测推翻**，被推翻的那条最值钱。

## 三条纪律

1. **每条期望都要能被推翻。** 一条「无论数据怎样都成立」的期望不是期望，
   是定义 —— 它占据一行位置，却永远不携带信息。写之前问：**什么数据会让它红？**
2. **算不出来的期望要单列**（`unmeasurable`），不许并进「成立」。
   数据缺失与结论成立在输出里长得一模一样。
3. **时序类期望的读数落进 `benchmark.json`**（不进版本摘要）——
   它的证据里嵌着微秒数，写进摘要会让同一份输入跑两次得到两个版本号。

## 求值函数表 `_EVALUATORS`

`acceptance.wiring_report` 会核对：**配置里声明的期望都有求值函数，
且没有孤儿求值函数** —— 写了函数忘了写配置，与没写这个函数在产物里长得一样。
"""

from __future__ import annotations

from typing import Any, Callable

from .objective import OBJECTIVES, is_independent

#: 期望名 → 求值函数。返回 `(verdict, observed, detail)`。
#: `verdict ∈ {held, refuted, unmeasurable}`。
_EVALUATORS: dict[str, Callable[[dict[str, Any]], tuple[str, Any, str]]] = {}


def _ev(name: str):
    def deco(fn):
        _EVALUATORS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------- 小工具
def _ops(p: dict[str, Any]) -> list[str]:
    return sorted(p["config.json"]["resolved"]["ops"])


def _space(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["space.json"]["per_op"][op]


def _mono(p: dict[str, Any], op: str, strategy: str = "reject") -> dict[str, Any]:
    for r in p["ladder.json"]["monotonicity"]:
        if r["op"] == op and r["strategy"] == strategy:
            return r
    raise KeyError(f"没有这个单调性行：{op}/{strategy}")


def _adv(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["adversarial.json"]["per_op"][op]


def _genes(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["genes.json"]["per_op"][op]


def _verdict(expect: bool, observed: bool) -> str:
    return "held" if bool(observed) == bool(expect) else "refuted"


# --------------------------------------------------------------------- 目标
@_ev("three_objectives_are_independent")
def _e_obj(p):
    """**逐算子**核：相关矩阵必须在该算子自己的空间上算。

    ⚠️ 第一版读的是 `space.json.objective_correlation` 这一个全局矩阵。
    把三个算子的候选**混在一起**算相关，量到的是「算子之间差多少」，
    不是「这个算子的三个目标轴冗不冗余」——
    而它印出来仍然是一个合法的相关系数（坑：口径与声明指的不是同一个对象）。
    """
    thr = float(p["config.json"]["resolved"]["search"].get(
        "objective_corr_threshold", 0.95))
    rows = [(op, _space(p, op)["objective_correlation"]) for op in _ops(p)]
    rows = [(op, c) for op, c in rows if c]
    if not rows:
        return "unmeasurable", None, "没有任何算子算出了相关矩阵"
    hit = [(op, is_independent(c, thr)) for op, c in rows]
    ok = all(v for _o, v in hit)
    st = {op: c.get("speedup", {}).get("throughput") for op, c in rows}
    return (_verdict(True, ok), ok,
            f"两两 |corr| < {thr} 的算子 {sum(1 for _o, v in hit if v)}/{len(hit)}；"
            f"corr(speedup, throughput) = {st} "
            f"（对固定算子这两轴**完全共线**，第三个目标轴是重复的）")


@_ev("tile_k_trades_accuracy_for_speed")
def _e_tilek(p):
    """`tile_k` 越大 ⇒ 块内规约树越深 ⇒ 更准；同时块边界更少 ⇒ 更快？

    判据分开看两件事：**精度方向**（tile_k ↑ ⇒ drift ↓）与**速度方向**
    （tile_k ↑ ⇒ latency ↓）。只要有一个方向不成立，这条期望就被推翻 ——
    而「不成立」本身是结论（说明这个权衡轴是单向的或不存在）。
    """
    rows: list[tuple[int, float, float]] = []
    for op in _ops(p):
        for r in _genes(p, op)["rows"]:
            if r["gene"] != "tile_k" or not r["measured"]:
                continue
            rows.append((op, r))
    if not rows:
        return "unmeasurable", None, "没有任何算子测到了 tile_k 的反事实配对"
    acc = []
    spd = []
    for op, r in rows:
        acc.append(r["marginal_drift_spread"] > 0)
        spd.append(r["marginal_latency_spread_us"] > 0)
    ok = any(acc) and any(spd)
    return (_verdict(True, ok), ok,
            f"tile_k 对精度有影响的算子 {sum(acc)}/{len(rows)}，"
            f"对延迟有影响的 {sum(spd)}/{len(rows)}")


@_ev("num_warps_is_an_accuracy_gene")
def _e_warps(p):
    hit = []
    for op in _ops(p):
        g = _genes(p, op)
        hit.append("num_warps" in g["numerically_active_genes"])
    ok = all(hit)
    return (_verdict(True, ok), ok,
            f"num_warps 在 {sum(hit)}/{len(hit)} 个算子上是精度基因"
            f"（{'全部' if ok else '**不是全部**'}）")


@_ev("memory_layout_is_numerically_inert")
def _e_layout(p):
    hit = []
    for op in _ops(p):
        g = _genes(p, op)
        hit.append("memory_layout" in g["numerically_inert_genes"])
    ok = all(hit)
    return (_verdict(True, ok), ok,
            f"memory_layout 在 {sum(hit)}/{len(hit)} 个算子上**恰好**对误差无影响"
            "（它只改访存，不改加法树）")


# --------------------------------------------------------------------- 阶梯
@_ev("ladder_buys_speed")
def _e_ladder(p):
    """**本项目与 `L6-graphopt` 的分水岭。**

    那边放宽容差买到的质量是**零**（前沿一格没动）；这边必须买得到。
    判据：每个算子上，最松那一级的最优加速比 / 最紧那一级 − 1 ≥ 门槛。
    """
    thr = float(p["config.json"]["resolved"]["acceptance"]["min_ladder_speedup_gain"])
    rows = []
    for op in _ops(p):
        m = _mono(p, op)
        sp = [x for x in m["best_speedup_oracle"] if x is not None]
        if len(sp) < 2 or sp[0] <= 0:
            rows.append((op, None))
            continue
        rows.append((op, sp[-1] / sp[0] - 1.0))
    got = [r for _o, r in rows if r is not None]
    if not got:
        return "unmeasurable", None, "没有任何算子有可比的加速比序列"
    worst = min(got)
    ok = worst >= thr
    return (_verdict(True, ok), round(worst, 9),
            f"最差算子上的相对增益 {worst:.4f}（要求 ≥ {thr}）；"
            f"明细 {[(o, None if v is None else round(v, 4)) for o, v in rows]}")


@_ev("safe_tolerance_window_is_nonempty")
def _e_window(p):
    """存在一个 τ：全部**合法**精度通过、全部**非法**扰动被拒。

    ⚠️ 这里的「全部合法精度」指**整个合法空间**的漂移上界，
    不是基线自己（基线漂移恒为 0，会让窗口的下界算成 0）。
    """
    rows = [(op, _adv(p, op)["safe_window"]["nonempty"]) for op in _ops(p)]
    ok = all(v for _o, v in rows)
    detail = "；".join(
        f"{op}: 合法最差 {_adv(p, op)['safe_window']['legal_max_drift']:.3e} "
        f"vs 非法最好 {_adv(p, op)['safe_window']['illegal_min_drift']:.3e}"
        for op in _ops(p))
    return (_verdict(True, ok), ok,
            f"窗口非空 {sum(1 for _o, v in rows if v)}/{len(rows)} 个算子；{detail}")


@_ev("precision_gate_is_needed")
def _e_gate2(p):
    n = sum(len(_adv(p, op)["gate1_only_anywhere"]) for op in _ops(p))
    ok = n >= 1
    names = {op: _adv(p, op)["gate1_only_anywhere"] for op in _ops(p)}
    return (_verdict(True, ok), n,
            f"门 1 放行、门 2 拒掉的非法 kernel 共 {n} 个"
            f"（{'存在' if ok else '**一个都没有** —— 门 2 是装饰'}）；{names}")


@_ev("effective_bits_has_a_sqrt_n_bias")
def _e_bits(p):
    """实测有效位宽 ≈ 声明位宽 − 0.5·log2(累加步数)（在 1 位以内）。

    ⚠️ 只核**合法** kernel。非法变体的有效位宽与它的声明**故意**不符 ——
    那正是门 2 要抓的东西，把它算进来会让这条期望永远不成立。
    """
    worst = 0.0
    n = 0
    who = None
    for op in _ops(p):
        for t in p["space.json"]["per_op"][op]["sample_trials"]:
            residual = t["effective_bits"] - (t["declared_bits"] - t["bits_bias"])
            if abs(residual) > worst:
                worst, who = abs(residual), f"{op}/{t['genome']['accum_dtype']}"
            n += 1
    if n == 0:
        return "unmeasurable", None, "没有可核的样本"
    ok = worst <= 1.0
    return (_verdict(True, ok), round(worst, 6),
            f"{n} 个样本上 |实测 − 解析预测| 最大 {worst:.4f} 位"
            f"（{who}，要求 ≤ 1.0）")


@_ev("bf16_is_free")
def _e_bf16(p):
    """在宽松容差下，最优可行 kernel 就是 bf16 且它**仍然诚实**。

    判据落在**头条对照表**上（`adversarial.json.per_op[op].headline`）：
    诚实的 bf16 那一行必须 `gate2_bits_ok`。少了这个对照，
    「bf16 又快又诚实」这句话就没有被任何数据支持。
    """
    hit = []
    for op in _ops(p):
        h = _adv(p, op).get("headline", {})
        ok = bool(h.get("available")) and bool(
            h.get("honest_bf16", {}).get("gate2_bits_ok"))
        hit.append((op, ok, h.get("available")))
    ok = all(v for _o, v, _a in hit)
    return (_verdict(True, ok), ok,
            f"bf16 既最快又诚实（门 2 放行）的算子 "
            f"{sum(1 for _o, v, _a in hit if v)}/{len(hit)}；"
            f"对照表可用性 {[(o, a) for o, _v, a in hit]}")


@_ev("default_tolerance_admits_all_precisions")
def _e_default(p):
    """`semantics.tolerance` 的默认值能收下全部三个精度档。"""
    dt_ok = []
    for op in _ops(p):
        sp = _space(p, op)
        counts = sp["feasible_by_dtype_at_default_tolerance"]
        dt_ok.append((op, all(v > 0 for v in counts.values()), counts))
    ok = all(v for _o, v, _c in dt_ok)
    return (_verdict(True, ok), ok,
            f"默认容差收下全部精度档的算子 {sum(1 for _o, v, _c in dt_ok if v)}"
            f"/{len(dt_ok)}；明细 "
            f"{[(o, c) for o, _v, c in dt_ok]}")


# --------------------------------------------------------------------- 搜索
@_ev("searchers_beat_random")
def _e_search(p):
    rows = []
    for op in _ops(p):
        for m in p["budget_sweep.json"]["per_op"][op]["margins"]:
            if m["budget"] == p["config.json"]["resolved"]["search"]["budget_per_searcher"]:
                rows.append((op, m["gain_over_random"], m["best_searcher"]))
    if not rows:
        return "unmeasurable", None, "预算扫描里没有主预算那一行"
    ok = all(g > 0 for _o, g, _s in rows)
    return (_verdict(True, ok), ok,
            f"主预算下非随机搜索器相对随机的增益 "
            f"{[(o, round(g, 6), s) for o, g, s in rows]}")


@_ev("random_saturates_at_main_budget")
def _e_sat(p):
    rows = [(op, p["budget_sweep.json"]["per_op"][op]["random_saturation_budget"])
            for op in _ops(p)]
    ok = all(v is not None for _o, v in rows)
    return (_verdict(True, ok), ok,
            f"随机搜索的饱和预算 {[(o, v) for o, v in rows]}"
            f"（None = 到主预算仍未饱和）")


@_ev("soft_constraint_changes_the_answer")
def _e_soft(p):
    diff = []
    for op in _ops(p):
        cells = p["ladder.json"]["per_op"][op]
        for tol in p["config.json"]["resolved"]["ladder"]["tolerances"]:
            a = cells.get(f"reject@{tol:g}", {}).get("cells", {}).get("oracle")
            b = cells.get(f"penalty@{tol:g}", {}).get("cells", {}).get("oracle")
            if not a or not b:
                continue
            ga = (a.get("best") or {}).get("genome_id")
            gb = (b.get("best") or {}).get("genome_id")
            if ga != gb:
                diff.append((op, tol))
    ok = bool(diff)
    return (_verdict(True, ok), len(diff),
            f"reject 与 penalty 交出不同答案的格子 {len(diff)} 个"
            f"（{'存在' if ok else '**一个都没有** —— 软约束没改变任何东西'}）")


# --------------------------------------------------------------------- 成本
@_ev("launch_overhead_dominates_the_speedup")
def _e_launch(p):
    """把 `launch_overhead_us` 扫到 0 后，最优 kernel 的加速比是否大幅缩小。"""
    rows = []
    for op in _ops(p):
        cs = p["benchmark.json"]["cost_sweep"].get(op)
        if not cs:
            continue
        s0 = next(r["speedup"] for r in cs["rows"] if r["launch_overhead_us"] == 0.0)
        smax = max(r["speedup"] for r in cs["rows"])
        rows.append((op, s0, smax, (smax - s0) / smax if smax else 0.0))
    if not rows:
        return "unmeasurable", None, "没有成本扫描数据"
    shrink = min(r[3] for r in rows)
    ok = shrink >= 0.2
    return (_verdict(True, ok), round(shrink, 6),
            f"把 launch 开销扫到 0 后加速比缩小的比例，最小 {shrink:.4f}"
            f"（≥0.2 才算「主要来自 launch」）；明细 "
            f"{[(o, round(a, 3), round(b, 3)) for o, a, b, _s in rows]}")


@_ev("analytic_latency_tracks_wallclock")
def _e_wall(p):
    """解析延迟与实测求值时间的排序同向（秩相关 ≥ 0.5）。

    ⚠️ **时序读数**：它的证据里嵌着微秒数 ⇒ 读数走 `benchmark.json`，
    这里只留结论与条数（见模块文档第 3 条）。
    """
    from .benchmark import spearman
    rows = []
    for op in _ops(p):
        w = p["benchmark.json"]["wallclock"].get(op)
        if not w or w["n"] < 3:
            continue
        xs = [r["analytic_latency_us"] for r in w["rows"]]
        ys = [r["measured_eval_us"] for r in w["rows"]]
        rho = spearman(xs, ys)
        rows.append((op, rho, w["n"]))
    if not rows:
        return "unmeasurable", None, "没有可核的实测样本"
    good = [r for _o, r, _n in rows if r == r and r >= 0.5]
    ok = len(good) == len(rows)
    return (_verdict(True, ok), round(min(r for _o, r, _n in rows), 6),
            f"秩相关 {[(o, None if r != r else round(r, 4), n) for o, r, n in rows]}"
            f"（要求全部 ≥ 0.5）")


@_ev("precision_gate_never_rejects_legal")
def _e_gate2_clean(p):
    n = 0
    for op in _ops(p):
        sp = _space(p, op)
        n += sp["n_legal_rejected_by_gate2"]
    ok = n == 0
    return (_verdict(True, ok), n,
            f"被门 2 误伤的合法 kernel 共 {n} 个"
            f"（{'零' if ok else '**非零 —— slack 标定得太紧**'}）")


# --------------------------------------------------------------------- 主入口
def run_expectations(products: dict[str, Any]) -> dict[str, Any]:
    declared = products["config.json"]["resolved"]["expectations"]
    items: list[dict[str, Any]] = []
    timing_items: list[dict[str, Any]] = []
    for name in sorted(declared):
        spec = declared[name]
        fn = _EVALUATORS.get(name)
        if fn is None:
            items.append({
                "name": name, "claim": spec["claim"], "expect": spec["expect"],
                "verdict": "unmeasurable", "observed": None,
                "detail": "**没有任何求值函数认领这条期望** —— "
                          "它看起来像一条设计承诺，其实没有人去核过",
            })
            continue
        try:
            verdict, observed, detail = fn(products)
        except Exception as e:                      # noqa: BLE001
            # ⚠️ 求值函数自己抛异常**必须记成 unmeasurable**，不许吞成 held。
            #    `L6-graphopt` 的坑 287：一个吞掉异常的兜底，等于一条静默失效的通道。
            verdict, observed, detail = (
                "unmeasurable", None, f"求值函数抛异常：{type(e).__name__}: {e}")
        row = {
            "name": name, "claim": spec["claim"], "expect": spec["expect"],
            "verdict": verdict, "observed": observed, "detail": detail,
        }
        if name == "analytic_latency_tracks_wallclock":
            timing_items.append(row)
        else:
            items.append(row)

    held = sum(1 for i in items if i["verdict"] == "held")
    refuted = sum(1 for i in items if i["verdict"] == "refuted")
    unmeas = sum(1 for i in items if i["verdict"] == "unmeasurable")
    return {
        "items": items,
        "timing_items": timing_items,
        "n": len(items),
        "n_held": held,
        "n_refuted": refuted,
        "n_unmeasurable": unmeas,
        "n_timing": len(timing_items),
        "refuted_names": [i["name"] for i in items if i["verdict"] == "refuted"],
        "note": "**被推翻的那条最值钱** —— 它是设计稿与现实不一致的地方。",
    }


def registered() -> dict[str, Any]:
    return dict(_EVALUATORS)


__all__ = ["run_expectations", "registered", "_EVALUATORS"]
