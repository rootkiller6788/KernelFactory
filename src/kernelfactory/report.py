"""报告渲染：**只吃产物字典**，不读运行期对象。

## 这条纪律买到两样

1. `visualize` 能从 `out/` 重建报告，一行都不重跑流水线；
2. 它证明的东西更强：从「流水线是确定的」变成「**产物足以重造报告**」。

配一条对抗用例（`invariants.renderers_are_pure_functions_of_the_products`）：
把一次 run 折成 JSON 往返 → 从它重建 → 再渲染，与直接渲染**逐位比**。
那条用例在 `L5-quantfactory` 抓出过 `render_html` 里 `len(run.products) - 2`
在产物还没填满时印出「其余 −2 个文件」——
**负数比报错更危险，因为它看起来像个数字。**

## Markdown 与 HTML 共用一份文档结构

`build_doc(products)` 产出 `Block` 列表，两种格式各自渲染它。
分两套实现的话，两份报告会慢慢长歪，而没有任何地方会红。
"""

from __future__ import annotations

import html as _html
from typing import Any, Sequence

Block = tuple

#: 目标轴的展示名。
_AXIS = {
    "speedup": "加速比",
    "throughput": "吞吐",
    "effective_bits": "有效位宽",
}


def _f(x: Any, nd: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x == float("inf"):
            return "∞"
        return f"{x:.{nd}f}"
    return str(x)


def _sci(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x == float("inf"):
            return "∞"
        return f"{x:.{nd}e}"
    return str(x)


def _pct(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _ops(p: dict[str, Any]) -> list[str]:
    return sorted(p["config.json"]["resolved"]["ops"])


def _space(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["space.json"]["per_op"][op]


def _genes(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["genes.json"]["per_op"][op]


def _adv(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["adversarial.json"]["per_op"][op]


def _ladder(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["ladder.json"]["per_op"][op]


def _mono(p: dict[str, Any], op: str, strategy: str) -> dict[str, Any]:
    return next(r for r in p["ladder.json"]["monotonicity"]
                if r["op"] == op and r["strategy"] == strategy)


# --------------------------------------------------------------------- 文档结构
def build_doc(products: dict[str, Any]) -> list[Block]:
    cfg = products["config.json"]["resolved"]
    base = products["baseline.json"]
    space = products["space.json"]
    genes = products["genes.json"]
    adv = products["adversarial.json"]
    ladder = products["ladder.json"]
    search = products["search.json"]
    sweep = products["budget_sweep.json"]
    bench = products["benchmark.json"]
    exp = products["expectations.json"]
    acc = products["acceptance.json"]
    summ = products["summary.json"]

    ops = _ops(products)
    tolerances = [float(x) for x in cfg["ladder"]["tolerances"]]
    tau0 = min(tolerances)

    b: list[Block] = []

    # ---------------------------------------------------------------- 1 概览
    b.append(("h", 1, "KernelFactory — 运行报告"))
    b.append(("p",
              "形状判定：**搜索闭环 · 硬约束 · 第三形态（数值容差）**。"
              "候选是 **KernelGenome**（9 个基因），硬约束是"
              "**「这个 kernel 的输出与基线一致」**，阶梯是**一致性容差 τ 逐级放松**。"
              "前两形态是 `L5-modeloptimizationfactory`（资源轴）与 "
              "`L6-graphoptimizationfactory`（语义等价）。"))
    b.append(("p",
              "⚠️ **本机没有可用 GPU，也没有 torch / triton / numpy。**"
              "交付的是一个**数值孪生**：误差是**真跑**出来的"
              "（分块累加 + 按位舍入，用 `frexp`/`ldexp` 手工实现 tf32/bf16 的舍入），"
              "延迟是**解析算**出来的（`cost.py`，每个常数都在 "
              "`baseline.json.cost_source` 里标了 `assumption` / `derived`）。"
              "**把假设标成测量是这类报告最危险的失真**，所以两个形状都印在产物里。"))
    b.append(("table", ["项", "值"], [
        ["协议指纹", products["config.json"]["protocol_digest"]],
        ["算子", "、".join(f"`{o}`" for o in ops)],
        ["基因", f"{len(cfg['genome'])} 个"
                 f"（README 原稿 8 个 + `accum_dtype`）"],
        ["搜索空间（逐算子）",
         "；".join(f"{o}: {_space(products, o)['n_genomes']}" for o in ops)],
        ["搜索器", "、".join(search["searchers"])],
        ["预算", f"oracle = 空间大小 / 其余 = "
                 f"{search['budget_per_searcher']}"],
        ["约束策略", "、".join(["reject", "penalty"])],
        ["阶梯容差", "、".join(f"{t:g}" for t in tolerances)],
        ["默认容差", f"{cfg['semantics']['tolerance']:g}"
                     "（**这是「未测量时会选的容差」，不是推荐值**）"],
        ["门 2 slack", f"{cfg['numerics']['precision_slack_bits']:g} 位"
                       "（标定出来的）"],
        ["缓存（唯一 / 命中）", f"{search['cache']['unique_evaluations']} / "
                               f"{search['cache']['cache_hits']}"
                               f"（命中率 {_f(search['cache']['hit_rate'], 4)}）"],
    ]))

    # ---------------------------------------------------------------- 2 核心命题
    b.append(("h", 2, "核心命题：**一个门槛不够**"))
    b.append(("p",
              "README 的命题是 `K* = arg min_K Latency(K) subject to Error(K) < ε`。"
              "**这句话里只有一个门槛，而它不够。**"
              "本项目的第二道门是：**声明与实现是否一致**。"))
    b.append(("table", ["门", "判据", "量的是什么"], [
        ["门 1｜一致性",
         "`drift(K) = ‖K − 基线‖ ≤ τ`",
         "输出像不像基线 —— 与 τ 有关"],
        ["门 2｜声明",
         "`effective_bits(K) ≥ declared_bits(K) − √n 偏置 − slack`",
         "**声明与实现是否一致** —— 与 τ **无关**"],
    ]))
    b.append(("p",
              "**交付判据 = 门 1 ∧ 门 2。**"
              "下面这张对照表是整份报告最想印的东西 —— "
              "对照做在**最难区分的那一对**上（诚实 bf16 里漂移最接近假 fp32 的那个）："))
    b.append(("table",
              ["算子", "行", "声明", "drift", "门 1 @τ=1e-2", "门 2", "有效位宽"],
              [row for op in ops for row in _headline_rows(products, op)]))
    ratios = summ["headline_drift_ratio"]
    b.append(("p",
              "**两行的漂移同量级、门 1 的结论相同，而门 2 是唯一分开它们的判据。**"
              f"漂移比（假装 fp32 ÷ 最难区分的诚实 bf16）："
              + "；".join(f"`{o}` = **{_f(r, 6)}**" for o, r in sorted(ratios.items()))
              + "。**比值 ≈ 1 意味着两行在误差上几乎无法区分** —— "
              "在 layernorm 上两者**逐位相同**（比值恰好 1.0，见 `bit_identical_drift`）。"
              "一个只报误差的验收表会把这两行混为一谈，而它看起来完全正常。"))
    b.append(("p",
              "⚠️ **诚实 bf16 有两个代表，它们回答两个不同的问题**："
              "「漂移最大的那个诚实 bf16」定义**上界**；"
              "而门 1 / 门 2 的对照必须做在「漂移最接近假 fp32 的那个」上。"
              "拿最差的诚实 bf16 去比一个具体的假 fp32，量到的是"
              "「这两个 kernel 差多少」，**不是**「假 fp32 混不混得进去」——"
              "本项目第一版就是这么错的（matmul 上直接红了）。"))
    b.append(("p",
              "假 fp32 的漂移落在**全部诚实 bf16 的漂移区间之内**："
              + "；".join(
                  f"`{o}` = {summ['headline_fake_inside_honest_range'].get(o)}"
                  f"（诚实区间 [{_sci(_adv(products, o)['headline']['honest_bf16_range']['min_drift'])}"
                  f", {_sci(_adv(products, o)['headline']['honest_bf16_range']['max_drift'])}]）"
                  for o in ops if o in summ["headline_fake_inside_honest_range"])
              + "。**这一条是可被推翻的** —— 若那个变体实际按 2 位跑，"
              "它的漂移（≈0.25）会远大于任何诚实 bf16，于是落在区间之外，"
              "而「门 1 分不开它们」这个论断当场失效。"))

    # ---------------------------------------------------------------- 3 空间
    b.append(("h", 2, "搜索空间"))
    b.append(("table",
              ["算子", "原始", "被过滤", "合法", "折叠重复", "不同目标向量",
               "精度档", "不同有效位宽"],
              [[op, str(_space(products, op)["n_raw"]),
                str(_space(products, op)["n_pruned"]),
                str(_space(products, op)["n_genomes"]),
                str(_space(products, op)["n_collapsed_duplicates"]),
                str(_space(products, op)["n_distinct_objective_vectors"]),
                str(_space(products, op)["n_distinct_accum_dtypes"]),
                str(_space(products, op)["n_distinct_effective_bits"])]
               for op in ops]))
    b.append(("p",
              "被过滤掉的原因分布 —— **一个「空间比预期小」的读数，"
              "与一个「过滤规则写错、砍掉了半个空间」的读数，"
              "在只看大小时无法区分**："))
    for op in ops:
        reasons = _space(products, op)["prune_reasons"]
        b.append(("p", f"`{op}`：" + "；".join(
            f"{k} = {v}" for k, v in sorted(reasons.items()))))
    b.append(("p",
              "**目标轴其实是两条。**"
              "`speedup` 与 `throughput` 对固定算子完全共线（相关系数恰好 1.0）——"
              "这是 README 那三条读数里的一个真实冗余。"
              "第三轴 `effective_bits` 与第一轴的关系是**真权衡**"
              "（bf16 快 4 倍但只有 8 位有效）。"))
    b.append(("table", ["算子", "corr(speedup, throughput)",
                        "corr(speedup, effective_bits)"],
              [[op,
                _f(_space(products, op)["objective_correlation"]
                   .get("speedup", {}).get("throughput"), 6),
                _f(_space(products, op)["objective_correlation"]
                   .get("speedup", {}).get("effective_bits"), 6)]
               for op in ops]))

    # ---------------------------------------------------------------- 4 基线
    b.append(("h", 2, "手写基线"))
    b.append(("p", "基线是**所有 drift 的比较基准**（它自己的漂移恰好 0），"
                   "同时它的指标定义了 `speedup` 的分母。"
                   "它刻意取空间里最保守的一档，于是有充足的优化余量。"))
    b.append(("table",
              ["算子", "基线基因组", "探测形状", "产出形状", "延迟 µs",
               "瓶颈", "效率"],
              [[op, base["per_op"][op]["label"],
                "×".join(str(x) for x in base["per_op"][op]["probe_shape"]),
                "×".join(str(x) for x in base["per_op"][op]["production_shape"]),
                _f(base["per_op"][op]["metrics"]["latency_us"], 3),
                base["per_op"][op]["metrics"]["bottleneck"],
                _f(base["per_op"][op]["metrics"]["efficiency"], 4)]
               for op in ops]))
    b.append(("p", "⚠️ **误差在探测形状上真跑，延迟在产出形状上解析算** —— "
                   "两个形状刻意不同。本机没有 GPU，在 512³ 上真跑一遍 Python 浮点"
                   "要几分钟。把假设标成测量是这类报告最危险的失真，"
                   "所以两个形状都进产物（有一条对抗用例核它们必须不同）。"))
    b.append(("p", "成本常数的来源标签："
              + "；".join(f"`{k}`={v['kind']}"
                          for k, v in base["cost_source"].items())))

    # ---------------------------------------------------------------- 5 阶梯
    b.append(("h", 2, "阶梯：容差逐级放松"))
    b.append(("p", "**这是本项目与 `L6-graphopt` 的分水岭。**"
                   "那边「语义等价」这条约束**完全不 binding**（放宽容差买到的质量是"
                   "**零**，前沿一格没动）；这边约束**真的 binding** ——"
                   "放宽容差真的买到速度。"
                   "`active_steps` 就是那个读数：**一个全平的 `active_steps` "
                   "在这里意味着精度轴没接进成本模型，是 bug 不是结论。**"))
    for op in ops:
        b.append(("h", 3, f"算子 `{op}`"))
        for strat in ("reject", "penalty"):
            rows = sorted([r for r in _ladder(products, op).values()
                           if r["strategy"] == strat],
                          key=lambda r: r["tolerance"])
            b.append(("table",
                      ["τ", "可行空间", "oracle 交付集", "oracle 前沿点",
                       "不同权衡点", "oracle 超体积", "最好加速比"],
                      [[f"{r['tolerance']:g}", str(r["n_feasible_space"]),
                        str(r["cells"]["oracle"]["n_deliverable"]),
                        str(r["cells"]["oracle"]["n_frontier"]),
                        str(r["cells"]["oracle"]["n_frontier_distinct"]),
                        _f(r["cells"]["oracle"]["hypervolume"], 6),
                        _f((r["cells"]["oracle"]["best"] or {}).get("speedup"), 4)]
                       for r in rows]))
            mo = _mono(products, op, strat)
            b.append(("p",
                      f"单调性（{strat}）："
                      + ("**单调不减** —— τ 放松 ⇒ 可行域扩张 ⇒ 上确界不减，"
                         "这是数学必然，红了就是实现错了。"
                         if mo["monotone"] else
                         f"**出现下降**：feasible={mo['monotone_feasible']} "
                         f"hv={mo['monotone_hypervolume']} "
                         f"latency={mo['monotone_latency']}")
                      + f"（`feasible_steps={mo['feasible_steps']}`、"
                        f"`active_steps={mo['active_steps']}`）"))
        sp = _mono(products, op, "reject")["best_speedup_oracle"]
        spn = [x for x in sp if x is not None]
        if len(spn) >= 2 and spn[0] > 0:
            b.append(("p",
                      f"**阶梯买到速度**：最紧那一级的 oracle 最优加速比 "
                      f"**{_f(spn[0], 4)}×** → 最松那一级 **{_f(spn[-1], 4)}×**"
                      f"（相对增益 **{_pct(spn[-1] / spn[0] - 1.0)}**）。"
                      "这就是「硬约束真的 binding」的证据。"))

    # ---------------------------------------------------------------- 6 搜索器
    dtol = float(cfg["semantics"]["tolerance"])
    b.append(("h", 2, f"搜索器对比（默认容差 τ={dtol:g} / reject）"))
    b.append(("p", "⚠️ **量的是「交出的答案的延迟」，不是「前沿覆盖度」。**"
                   "greedy 沿**一个**标量化目标爬山，所以它只覆盖前沿上的**一个**点 —— "
                   "用超体积去量它，量到的是「它的设计目标不是这个」，不是「它弱」。"
                   "两种读数都印在下面。"))
    for op in ops:
        cells = _ladder(products, op)[f"reject@{dtol:g}"]["cells"]
        ob = cells["oracle"]["best"]
        rows: list[list[str]] = []
        for n, c in sorted(cells.items()):
            if ob is None or c["best"] is None:
                gap = "—"
            elif n == "oracle":
                gap = "0（它就是 oracle）"
            else:
                gap = _pct(max(0.0, c["best"]["latency_us"] / ob["latency_us"] - 1.0))
            rows.append([
                n, str(c["budget_used"]), str(c["n_feasible"]),
                str(c["n_deliverable"]), str(c["n_frontier"]),
                str(c["n_frontier_distinct"]),
                _f(c["hypervolume"], 6), gap,
                _f((c["best"] or {}).get("speedup"), 4),
            ])
        b.append(("h", 3, f"算子 `{op}`"))
        b.append(("table",
                  ["搜索器", "实际用", "可行", "交付", "前沿点", "不同权衡点",
                   "超体积", "相对 oracle 的**答案延迟**缺口", "最好加速比"],
                  rows))
    b.append(("h", 3, "前沿覆盖度随 τ 变化（**为什么它不当门槛**）"))
    b.append(("p", "下面每格是「非随机搜索器里最好的那个」相对 oracle 的**超体积**缺口。"
                   "τ 越松、前沿越宽，标量化搜索器能覆盖的比例就越低 —— "
                   "**这是它的设计目标决定的，不是能力问题。**"))
    b.append(("table",
              ["算子", "策略", "τ"] + list(cfg["search"]["searchers"]),
              [[op, strat, f"{t:g}"]
               + [_pct(max(0.0, (o - _ladder(products, op)[f"{strat}@{t:g}"]
                          ["cells"][n]["hypervolume"]) / o)) if o > 0 else "—"
                  for n in cfg["search"]["searchers"]]
               for op in ops for strat in ("reject",)
               for t in tolerances
               for o in [_ladder(products, op)[f"{strat}@{t:g}"]
                         ["cells"]["oracle"]["hypervolume"]]]))
    b.append(("p",
              "**「前沿点」与「不同权衡点」是两个数，都要报。**"
              "按点数报前沿大小会被**共线的第三轴**灌水：`throughput` 与 `speedup` "
              "完全共线，同一个权衡点会被算两次。门槛落在去重后的个数上。"))
    b.append(("p",
              "**预算是不对称的，而且必须显式报出来**：oracle 拿整个空间"
              "（逐算子不同），其余拿 "
              f"{search['budget_per_searcher']}。这不是特权，是定义 —— "
              "若把 oracle 也压在同一个预算，它交出的就是「那个预算里的最优」，"
              "而它就不再是 oracle，`oracle_is_optimal` 会从数学必然退化成「可能」。"))

    # ------------------------------------------------------------- 6b 预算扫描
    b.append(("h", 3, "预算扫描：**单一预算点不是比较，是轶事**"))
    b.append(("p", f"主预算 {cfg['search']['budget_per_searcher']}、"
                   f"紧预算 {cfg['search']['tight_budget']}。"
                   "随机搜索在某个预算上就把 oracle 的超体积拿满了 —— "
                   "**在那里比策略等于比运气**。"))
    for op in ops:
        sw = sweep["per_op"][op]
        b.append(("table",
                  ["预算"] + list(sw["searchers"]) + ["oracle"],
                  [[str(r["budget"])]
                   + [_f(r["hypervolume"][n], 6) for n in sw["searchers"]]
                   + [_f(sw["oracle_hypervolume"], 6)]
                   for r in sw["rows"]]))
        b.append(("p", f"`{op}`：随机搜索的饱和预算 = "
                  f"**{sw['random_saturation_budget']}**"
                  f"（None = 到最大预算仍未饱和）。"
                  "策略的贡献与预算的贡献是**耦合的** —— "
                  "只报一个预算点，等于替读者选了「在哪个预算下比较」这个前提。"))

    # ---------------------------------------------------------------- 7 基因
    b.append(("h", 2, "逐基因效应：**哪个基因值钱，哪个是装饰**"))
    b.append(("p",
              "手法是**反事实配对**：把某一个基因换成它的另一档，其余不动，"
              "看延迟与漂移变了多少。"
              "⚠️ **「测了发现没影响」与「根本没测到」必须分开报** —— "
              "一个基因可能没有任何反事实配对（所有替换都落到非法区），"
              "那时 `max_delta = 0`，而它读起来与「这个基因完全没用」一模一样。"))
    for op in ops:
        g = _genes(products, op)
        b.append(("h", 3, f"算子 `{op}`"))
        b.append(("table",
                  ["基因", "取值数", "配对", "最大 Δ延迟 µs", "最大 Δ漂移",
                   "边际 Δ延迟", "边际 Δ漂移", "数值惰性", "备注"],
                  [[r["gene"], str(r["n_values"]), str(r["n_pairs"]),
                    _f(r["max_latency_delta_us"], 4),
                    _sci(r["max_drift_delta"]),
                    _f(r["marginal_latency_spread_us"], 4),
                    _sci(r["marginal_drift_spread"]),
                    "**恰好为零**" if r["numerically_inert"]
                    else ("—" if not r["measured"] else "否"),
                    r["note"] or ""]
                   for r in g["rows"]]))
        b.append(("p",
                  f"数值惰性：{g['numerically_inert_genes'] or '（无）'}；"
                  f"数值活跃：{g['numerically_active_genes'] or '（无）'}；"
                  f"**未测到**：{g['unmeasured_genes'] or '（无）'}"))
    b.append(("p", "`numerically_inert` 的判据是「全部反事实配对的漂移差**恰好为 0**」。"
                   "**「恰好」是关键**：一个 1e-18 的差会让它变成 active，"
                   "而 1e-18 在 24 位精度下是不可表示的。"))

    # ---------------------------------------------------------------- 8 对抗
    b.append(("h", 2, "对抗：容差崖与安全窗口"))
    b.append(("p",
              "⚠️ 这一节的表里是**非法变体** —— 它们是**被测量的实验对象**（数据），"
              "**不是对抗用例**。对抗用例在「验收」那一节，是**不可调**的不变量断言。"
              "**同一个词指了两个不同的东西，读者没法从报告内部发现。**"))
    for op in ops:
        a = _adv(products, op)
        w = a["safe_window"]
        b.append(("h", 3, f"算子 `{op}`"))
        b.append(("p",
                  f"安全窗口 = **[{_sci(w['lo'])}, {_sci(w['hi'])})**"
                  + (f"，宽 {_f(w['width_decades'], 2)} 个数量级"
                     if w["nonempty"] and w["width_decades"] is not None
                     else "")
                  + f"（下界 = 合法空间漂移最大值，见证 `{w['witness_legal']}`；"
                    f"上界 = 非法变体最小漂移，见证 `{w['witness_illegal']}`）。"
                  + ("**窗口非空**：存在一个 τ 同时「收下全部合法精度」"
                     "与「拒掉全部非法扰动」。" if w["nonempty"] else
                     "**窗口是空的** —— 合法精度与非法扰动在误差上重叠了。"
                     "**这是结论，不是 bug。**")))
        b.append(("table",
                  ["非法变体", "位数", "声明", "漂移", "有效位宽", "门 2"]
                  + [f"τ={t:g}" for t in tolerances],
                  [[r["variant"],
                    "—" if r["mantissa_bits"] is None else str(r["mantissa_bits"]),
                    r["declared_dtype"],
                    _sci(r["drift"]),
                    _f(r["effective_bits"], 3),
                    "过" if r["gate2_bits_ok"] else "**拒**"]
                   + ["收" if r["accepted_at"][str(t)] else "·"
                      for t in tolerances]
                   for r in a["variants"]]))
        b.append(("p",
                  f"默认容差 τ={a['default_tolerance']:g} 下："
                  f"**门 1 单独会收下 {a['n_accepted_at_default_tolerance']} 个非法变体**"
                  f"（{a['accepted_at_default_tolerance'] or '（无）'}），"
                  f"其中门 2 拒掉了 **{a['n_gate1_only_at_default_tolerance']} 个**"
                  f"（{a['gate1_only_at_default_tolerance'] or '（无）'}）。"
                  "这不是 bug，是**配置取值的后果** —— "
                  "「校验器能不能挡住非法 kernel」这个问题的答案取决于容差取多少，"
                  "而容差是配置里的一个数。**不看容差的「校验器通过率」是没有意义的读数。**"))
        b.append(("p",
                  f"尺子网格横跨崖：**{a['ruler_grid_spans_cliff']}**"
                  f"（崖 = 合法实现能达到的最差精度 {_sci(w['legal_max_drift'])}；"
                  f"下方 {a['n_rulers_below_legal_max']} 档 / "
                  f"上方 {a['n_rulers_above_legal_max']} 档）。"
                  "少任何一侧，「合法低精度」与「偷偷降精度」在误差上重叠的"
                  "那一段就没被测量到 —— 而表看起来完全正常。"))
    b.append(("p", "**门 1 单独会放行的非法 kernel（跨全部算子、全部容差）**："
              + "；".join(f"`{op}` → {_adv(products, op)['gate1_only_anywhere'] or '（无）'}"
                          for op in ops)
              + "。这些就是**单门槛设计的代价**。"))

    # ---------------------------------------------------------------- 9 前沿
    fr = ladder["frontier"]
    for op in ops:
        if op not in fr:
            continue
        b.append(("h", 2, f"交付物：算子 `{op}` 在 τ={tau0:g} / reject 的 Pareto 前沿"))
        b.append(("p", "**交付物是前沿，不是点。**结果层绝不允许标量化 —— "
                       "一旦把三轴压成一个数，输出就不再是前沿。"
                       "下面的表只是把前沿**列出来**。"))
        pts = fr[op]["points"]
        b.append(("table",
                  ["基因组", "声明", "有效位宽", "漂移", "加速比", "吞吐",
                   "延迟 µs", "门 1", "门 2"],
                  [[p_["label"], f"{p_['declared_bits']} 位",
                    _f(p_["effective_bits"], 3), _sci(p_["drift"]),
                    _f(p_["speedup"], 4), _f(p_["throughput_tflops"], 4),
                    _f(p_["latency_us"], 3),
                    "过" if p_["feasible"] else "拒",
                    "过" if p_["bits_ok"] else "**拒**"]
                   for p_ in pts]))
        b.append(("svg", _frontier_svg(fr[op], op),
                  f"算子 {op} 的前沿在（加速比, 有效位宽）上的投影；"
                  "红叉 = 基线（1.0×, 24 位）"))
        b.append(("p", f"{fr[op]['n_frontier']} 个前沿点 / "
                       f"**{fr[op]['n_frontier_distinct']} 个不同权衡点**，"
                       f"超体积 {_f(fr[op]['hypervolume'], 6)}。"
                       "**这个投影就是那个真权衡**：bf16 在右边（快），"
                       "fp32 在上边（准），而两者不可兼得。"))
        best = ladder["best_answer"].get(op)
        if best:
            b.append(("p",
                      f"**头条数字**（τ={tau0:g} / reject，oracle 按最短延迟挑）："
                      f"`{best['label']}`，加速比 "
                      f"**{_f(best['speedup'], 4)}×**，"
                      f"有效位宽 **{_f(best['effective_bits'], 3)}** 位"
                      f"（声明 {best['declared_bits']} 位、√n 偏置 "
                      f"{_f(best['bits_bias'], 3)} 位），"
                      f"漂移 {_sci(best['drift'])}，"
                      f"门 1 = {best['feasible']}、门 2 = {best['bits_ok']}。"))

    # ---------------------------------------------------------------- 10 期望
    b.append(("h", 2, "设计期望 vs 实测"))
    b.append(("p", f"共 {exp['n']} 条：**{exp['n_held']} 条成立、"
                   f"{exp['n_refuted']} 条被推翻、"
                   f"{exp['n_unmeasurable']} 条算不出来**。"
                   "**被推翻的那几条最值钱** —— 它们是「原来想错了什么」。"
                   f"另有 {exp['n_timing']} 条**依赖实测耗时**的期望，"
                   "它们的读数随机器变，所以不在这个文件里"
                   "（见 `benchmark.json` 的 `timing_expectations`）。"))
    b.append(("table", ["期望", "原期望", "实测", "裁定", "读数"], [
        [i["name"], _f(i["expect"]), _f(i["observed"]),
         {"held": "成立", "refuted": "**被推翻**",
          "unmeasurable": "算不出来"}[i["verdict"]], i["detail"]]
        for i in exp["items"]]))
    timing = bench.get("timing_expectations") or []
    if timing:
        b.append(("h", 3, "时序类期望（**不进版本摘要**）"))
        b.append(("p", "它们的读数来自实测耗时 —— 同一份输入跑两次会得到不同的数。"
                       "把它们放进版本摘要，版本号就失去意义。"))
        b.append(("table", ["期望", "原期望", "实测", "裁定", "读数"], [
            [i["name"], _f(i["expect"]), _f(i["observed"]),
             {"held": "成立", "refuted": "**被推翻**",
              "unmeasurable": "算不出来"}[i["verdict"]], i["detail"]]
            for i in timing]))

    # ---------------------------------------------------------------- 11 验收
    b.append(("h", 2, "验收：门槛（可调）"))
    b.append(("p", f"{acc['n']} 条门槛：**{acc['n_passed']} 通过、"
                   f"{acc['n_failed']} 未通过**。"
                   "门槛是**达标线** —— 阈值写在 `config.acceptance` 里，可以被调松。"
                   + ("" if acc["n_failed"] == 0
                      else f" 未通过：{acc['failed']}")))
    b.append(("table", ["门槛", "阈值", "实测", "结论", "说明"], [
        [c["name"], str(c["threshold"]), _f(c["observed"]),
         "通过" if c["passed"] else "**未通过**", c["detail"]]
        for c in acc["checks"]]))

    # 11b 对抗用例：**不可调**的不变量断言
    if acc.get("cases"):
        n_cases = int(acc["n_cases"])
        n_bad = int(acc["n_cases_failed"])
        b.append(("h", 2, "对抗用例（**不可调**的不变量断言）"))
        b.append(("p", f"{n_cases} 条：**{n_cases - n_bad} 成立、{n_bad} 不成立**。"
                       "门槛是**达标线**（阈值写在配置里，可以被调松）；"
                       "这一节是**不变量**（配置里没有它的阈值）——"
                       "红了说明代码或产物自相矛盾，不是「这次没达标」。"
                       + ("" if n_bad == 0
                          else f" 不成立的：{acc['failed_cases']}")))
        b.append(("table", ["对抗用例", "结论", "说明"], [
            [c["name"], "成立" if c["ok"] else "**不成立**", c["detail"]]
            for c in acc["cases"]]))
        b.append(("p", "⚠️ 口径：`adversarial.json` 里那些**非法变体**是"
                       "**被测量的实验对象**（数据），**不是**对抗用例。"
                       "两者混称会让「这个项目有多少条对抗用例」这句话没有意义。"))
    b.append(("p", "接线核对（**门槛键有没有被源码真的读过**）："
              + f"未接线 `{acc['wiring']['unwired_acceptance_keys']}`；"
                f"孤儿期望求值函数 `{acc['wiring']['orphan_expectation_evaluators']}`；"
                f"缺求值函数 `{acc['wiring']['missing_expectation_evaluators']}`；"
                f"声明了但配置里没有的超参 "
                f"`{acc['wiring']['searcher_params_not_in_config']}`。"))

    # ---------------------------------------------------------------- 12 成本
    b.append(("h", 2, "成本模型：假设与敏感度"))
    b.append(("p", "成本模型**全部是算出来的**（`cost_source` 逐键标注）。"
                   "下面扫 `launch_overhead_us`，一次只动一个变量 —— "
                   "凡「某机制买到 X」的结论，都要报告它在哪个系数区间才成立。"))
    for op in ops:
        cs = bench.get("cost_sweep", {}).get(op)
        if not cs:
            continue
        b.append(("table", ["launch_overhead_us", "基线延迟 µs", "最好延迟 µs",
                            "speedup", "best 仍然赢"],
                  [[_f(r["launch_overhead_us"], 3),
                    _f(r["baseline_latency_us"], 3),
                    _f(r["best_latency_us"], 3), _f(r["speedup"], 6),
                    _f(r["best_still_wins"])] for r in cs["rows"]]))
        b.append(("p", f"`{op}`：{cs['n_wins']}/{cs['n_rows']} 个取值下 best 仍然赢，"
                       f"speedup 区间 [{_f(cs['speedup_range'][0], 4)}, "
                       f"{_f(cs['speedup_range'][1], 4)}]。"
                       "**如果 best 只在部分取值下赢，那它赢的是这个常数，不是自己。**"))
    wc = bench.get("wallclock", {})
    b.append(("p", "解析延迟 vs **实测求值耗时**的秩相关（**这是孪生自身一致性的读数，"
                   "不是性能读数** —— 本机没有 GPU，「kernel 有多快」量不出来）："
              + "；".join(
                  f"`{op}` ρ = {_f(w.get('spearman_rho'), 6)}（{w.get('n')} 个样本）"
                  for op, w in sorted(wc.items()) if w.get("n"))))

    # ---------------------------------------------------------------- 13 产物
    b.append(("h", 2, "产物"))
    b.append(("p", "进版本摘要的产物（顺序即摘要顺序）："
              + "、".join(f"`{n}`" for n in products["version.json"]["versioned_files"])))
    b.append(("p", "落盘但不进摘要："
              + "、".join(f"`{n}`" for n in products["version.json"]["unversioned_files"])))
    b.append(("p", "摘要里刻意排除了三类：**自指**（`version.json`）、"
                   "**耗时**（`benchmark.json`）、**视图**（`report.*`）。"
                   "于是同一份输入跑两次：**只有 `version.json`（时间戳）与 "
                   "`benchmark.json`（耗时）不同，其余 "
                   f"{len(products['version.json']['versioned_files'])} 个结论性产物逐位相同**。"))

    # ---------------------------------------------------------------- 14 汇总
    b.append(("h", 2, "汇总"))
    b.append(("table", ["项", "值"], [
        ["版本", products["version.json"]["version"]],
        ["算子数", str(summ["n_ops"])],
        ["空间总计", f"{summ['space_size_total']}"
                     f"（{summ['space_size_per_op']}）"],
        ["头条答案的加速比",
         "；".join(f"{o}: {_f((summ['best_answer'].get(o) or {}).get('speedup'), 4)}×"
                   for o in ops)],
        ["阶梯买到速度（最松 ÷ 最紧）",
         "；".join(f"{o}: {_pct(_gain(summ['ladder_best_speedup_oracle'][o]))}"
                   for o in ops)],
        ["阶梯 active_steps（逐算子）", str(summ["ladder_active_steps_per_op"])],
        ["门 1 放行、门 2 拒掉（跨算子）", str(summ["n_gate1_only_anywhere"])],
        ["头条漂移比（假装 fp32 ÷ 诚实 bf16）", str(summ["headline_drift_ratio"])],
        ["安全窗口非空（逐算子）", str(summ["safe_window_nonempty"])],
        ["数值惰性基因（逐算子）", str(summ["numerically_inert_genes"])],
        ["未测到的基因（逐算子）", str(summ["unmeasured_genes"])],
        ["期望 成立/推翻/算不出",
         f"{summ['expectations']['n_held']} / {summ['expectations']['n_refuted']} / "
         f"{exp['n_unmeasurable']}"],
        ["验收 通过/总数",
         f"{summ['acceptance']['n_passed']} / {summ['acceptance']['n']}"],
        ["对抗用例 成立/总数",
         f"{summ['acceptance']['n_cases'] - (summ['acceptance']['n_cases_failed'] or 0)}"
         f" / {summ['acceptance']['n_cases']}"],
    ]))
    return b


def _gain(seq: Any) -> Any:
    if not isinstance(seq, list):
        return None
    sp = [x for x in seq if x is not None]
    if len(sp) < 2 or sp[0] <= 0:
        return None
    return sp[-1] / sp[0] - 1.0


def _headline_rows(products: dict[str, Any], op: str) -> list[list[str]]:
    h = _adv(products, op).get("headline", {})
    if not h.get("available"):
        return [[op, "（对照表不可用）", str(h.get("why", "")), "—", "—", "—", "—"]]
    n = h["nearest_honest_bf16"]
    f = h["fake_fp32"]
    w = h["honest_bf16"]
    return [
        [op, "诚实的 bf16（最难区分的那一个）", n["declared_dtype"],
         _sci(n["drift"]),
         "过" if n["gate1_feasible_at_1e-2"] else "拒",
         "过" if n["gate2_bits_ok"] else "**拒**",
         _f(n["effective_bits"], 3)],
        [op, "**声称 fp32、实际 8 位**", f["declared_dtype"], _sci(f["drift"]),
         "过" if f["gate1_feasible_at_1e-2"] else "拒",
         "**拒**" if not f["gate2_bits_ok"] else "过",
         _f(f["effective_bits"], 3)],
        [op, "（对照）漂移最大的诚实 bf16", w["declared_dtype"],
         _sci(w["drift"]),
         "过" if w["gate1_feasible_at_1e-2"] else "拒",
         "过" if w["gate2_bits_ok"] else "**拒**",
         _f(w["effective_bits"], 3)],
    ]


# --------------------------------------------------------------------- 前沿 SVG
def _frontier_svg(fr: dict[str, Any], op: str, size: int = 190) -> str:
    pts = fr.get("points", [])
    keys = ["speedup", "effective_bits"]
    labels = [_AXIS[k] for k in keys]
    pad = 40
    w = pad * 2 + size
    h = size + pad * 2
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">']
    xs = [p_["speedup"] for p_ in pts]
    ys = [p_["effective_bits"] for p_ in pts]
    lo = [min(xs) - 0.05 * max(1e-9, max(xs) - min(xs)) if pts else 0,
          min(ys) - 0.05 * max(1e-9, max(ys) - min(ys)) if pts else 0]
    hi = [max(xs) + 0.05 * max(1e-9, max(xs) - min(xs)) if pts else 1,
          max(ys) + 0.05 * max(1e-9, max(ys) - min(ys)) if pts else 1]
    ox, oy = pad, pad

    def sx(x: float) -> float:
        span = (hi[0] - lo[0]) or 1.0
        return ox + (x - lo[0]) / span * size

    def sy(y: float) -> float:
        span = (hi[1] - lo[1]) or 1.0
        return oy + size - (y - lo[1]) / span * size

    parts.append(f'<rect x="{ox}" y="{oy}" width="{size}" height="{size}" '
                 f'fill="#fbfbfd" stroke="#c9ccd6" stroke-width="1"/>')
    parts.append(f'<text x="{ox + size / 2:.0f}" y="{oy + size + 16}" '
                 f'text-anchor="middle" font-size="10" fill="#333">{labels[0]}</text>')
    parts.append(f'<text x="{ox - 8}" y="{oy + size / 2:.0f}" text-anchor="middle" '
                 f'font-size="10" fill="#333" '
                 f'transform="rotate(-90 {ox - 8} {oy + size / 2:.0f})">'
                 f'{labels[1]}</text>')
    for p_ in pts:
        parts.append(f'<circle cx="{sx(p_["speedup"]):.2f}" '
                     f'cy="{sy(p_["effective_bits"]):.2f}" r="3.4" '
                     f'fill="#2f6fd0" fill-opacity="0.85"/>')
    # 基线点 (1.0, 声明位宽 24) 的十字
    parts.append(f'<path d="M {sx(1.0) - 4:.2f} {sy(24.0):.2f} '
                 f'L {sx(1.0) + 4:.2f} {sy(24.0):.2f} '
                 f'M {sx(1.0):.2f} {sy(24.0) - 4:.2f} '
                 f'L {sx(1.0):.2f} {sy(24.0) + 4:.2f}" '
                 f'stroke="#c0392b" stroke-width="1.4" fill="none"/>')
    parts.append('</svg>')
    return "".join(parts)


# --------------------------------------------------------------------- 渲染器
def to_markdown(doc: Sequence[Block]) -> str:
    out: list[str] = []
    for blk in doc:
        kind = blk[0]
        if kind == "h":
            out.append("#" * blk[1] + " " + blk[2])
        elif kind == "p":
            out.append(blk[1])
        elif kind == "ul":
            out.extend(f"- {x}" for x in blk[1])
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append("| " + " | ".join(str(x) for x in headers) + " |")
            out.append("|" + "|".join(" --- " for _ in headers) + "|")
            for r in rows:
                out.append("| " + " | ".join(str(x) for x in r) + " |")
        elif kind == "svg":
            out.append(f"> （图形见 `report.html`：{blk[2]}）")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_html(doc: Sequence[Block], title: str) -> str:
    body: list[str] = []
    for blk in doc:
        kind = blk[0]
        if kind == "h":
            body.append(f"<h{blk[1]}>{_inline(blk[2])}</h{blk[1]}>")
        elif kind == "p":
            body.append(f"<p>{_inline(blk[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in blk[1])
                        + "</ul>")
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            th = "".join(f"<th>{_inline(x)}</th>" for x in headers)
            tr = "".join("<tr>" + "".join(f"<td>{_inline(x)}</td>" for x in r)
                         + "</tr>" for r in rows)
            body.append(f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody>"
                        "</table>")
        elif kind == "svg":
            body.append(f"<figure>{blk[1]}<figcaption>{_inline(blk[2])}</figcaption>"
                        "</figure>")
    css = """
    :root { color-scheme: light; }
    body { margin: 0 auto; max-width: 1240px; padding: 32px 24px 64px;
           background: #ffffff; color: #1c1f26;
           font: 14px/1.7 system-ui, -apple-system, "Segoe UI", sans-serif; }
    h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
    h2 { font-size: 19px; margin: 34px 0 10px; padding-bottom: 6px;
         border-bottom: 1px solid #e6e8ee; }
    h3 { font-size: 15px; margin: 22px 0 8px; color: #333a46; }
    p { margin: 8px 0; }
    code { background: #f2f4f8; border-radius: 4px; padding: 1px 5px;
           font: 12.5px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }
    table { border-collapse: collapse; margin: 12px 0 18px; width: 100%;
            font-size: 13px; }
    th, td { border: 1px solid #e2e5ec; padding: 6px 9px; text-align: left;
             vertical-align: top; }
    th { background: #f6f8fb; font-weight: 600; }
    tbody tr:nth-child(even) { background: #fcfdff; }
    figure { margin: 14px 0 22px; }
    figcaption { font-size: 12.5px; color: #5a6070; margin-top: 6px; }
    svg { max-width: 100%; height: auto; }
    strong { font-weight: 650; }
    """
    return ("<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_html.escape(title)}</title><style>{css}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")


def _inline(text: str) -> str:
    """极小的行内标记：`**粗体**`、`` `代码` ``。先转义再替换。"""
    s = _html.escape(str(text))
    while "**" in s:
        a = s.find("**")
        b = s.find("**", a + 2)
        if b < 0:
            break
        s = s[:a] + "<strong>" + s[a + 2:b] + "</strong>" + s[b + 2:]
    while "`" in s:
        a = s.find("`")
        b = s.find("`", a + 1)
        if b < 0:
            break
        s = s[:a] + "<code>" + s[a + 1:b] + "</code>" + s[b + 1:]
    return s


def render_markdown(products: dict[str, Any]) -> str:
    return to_markdown(build_doc(products))


def render_html(products: dict[str, Any]) -> str:
    v = products["version.json"]["version"]
    return to_html(build_doc(products), f"KernelFactory 运行报告 {v}")


__all__ = ["build_doc", "to_markdown", "to_html", "render_markdown",
           "render_html", "_frontier_svg"]
