"""对抗：注入**已知幅度**的非法 kernel，看两道门各自放行了什么。

## 这一节要证明的一件事

**门 1（`Error < ε`）不足以判定「这个 kernel 是对的」。**

证据不是一段论述，是一张表：把「诚实的 bf16」与「声称 fp32、实际按 8 位跑」
放在一起，两者的漂移**同量级**、门 1 的结论**相同**，
而它们的区别**不在误差里**，在**声明与实现是否一致**里。

## 两类非法 kernel

1. **具名 bug**：`drop_tail_rows` / `skip_normalization` / `no_centering` /
   `eps_zero` / `drop_tail` —— 它们各自对应一种真实实现错误。
2. **尺子** `mantissa_truncate(b)`：声称 fp32，实际按 `b` 位有效跑。
   漂移 ≈ 2^−b ⇒ **崖的位置是一个可解析定位的数**，
   而不是「随机破坏得到的、没有量纲的结论」。

⚠️ **这两类都是「被测量的实验对象」（数据），不是「对抗用例」。**
对抗用例是 `invariants.py` 里那些不可调的不变量断言 ——
第一版把两者混称，于是报告里写「7 条对抗用例」而实际是数据。
**同一个词指了两个不同的东西，读者没法从报告内部发现。**

## ε 网格必须横跨崖的两侧（这条断言只能在这里做）

配置校验只看得到位数与容差两个数、量不出漂移，所以它**不断言**两者的大小关系
（在 `config.py` 里写明了原因）。真正的断言在这里：漂移算出来之后，
尺子必须在「全部被接受」与「全部被拒」之间至少跨过一次。
"""

from __future__ import annotations

from typing import Any, Sequence

from .evaluate import KernelEvaluator, accum_steps, baseline_genome
from .numerics import SIGNIFICANT_BITS, effective_bits, expected_bits_bias
from .ops import ILLEGAL_VARIANTS, kernel_output
from .ops import drift_of


def _row(cfg: dict[str, Any], ev: KernelEvaluator, name: str,
         out: list[float], *, epsilon: int | None, declared: str,
         gen: Any, tolerances: Sequence[float], floor: float,
         slack: float) -> dict[str, Any]:
    drift = drift_of(out, ev.baseline_out, floor)
    err_ref = drift_of(out, ev.num.ref, floor)
    exact = ev.num.exact(gen)
    err_exact = drift_of(out, exact, floor)
    eff = effective_bits(err_exact)
    bias = expected_bits_bias(accum_steps(cfg, ev.op, gen))
    declared_bits = SIGNIFICANT_BITS[declared]
    headroom = eff - (declared_bits - bias)
    bits_ok = headroom >= -slack
    accepted = {str(t): bool(drift <= t) for t in tolerances}
    gate1 = [t for t in tolerances if drift <= t]
    return {
        "op": ev.op,
        "variant": name,
        "kind": "illegal",
        "mantissa_bits": epsilon,
        "declared_dtype": declared,
        "declared_bits": declared_bits,
        "drift": round(drift, 15),
        "error_vs_reference": round(err_ref, 15),
        "effective_bits": round(eff, 9),
        "bits_bias": round(bias, 9),
        "bits_headroom": round(headroom, 9),
        "gate2_bits_ok": bool(bits_ok),
        "accepted_at": accepted,
        "cliff_tolerance": (min(gate1) if gate1 else None),
        "accepted_any": bool(gate1),
        #: **本项目的核心读数**：门 1 放行、门 2 拒掉 ⇒ 单门槛设计会收下这个 bug。
        "gate1_only_would_accept": bool(gate1) and not bits_ok,
        "note": "",
    }


def run_adversarial(cfg: dict[str, Any], ev: KernelEvaluator,
                    tolerances: Sequence[float]) -> dict[str, Any]:
    floor = float(cfg["numerics"]["error_floor"])
    slack = float(cfg["numerics"]["precision_slack_bits"])
    gen = ev.baseline
    rows: list[dict[str, Any]] = []

    # ---- 尺子：mantissa_truncate(b) ----
    rulers: list[dict[str, Any]] = []
    for b in [int(x) for x in cfg["adversarial"]["mantissa_bits"]]:
        out = kernel_output(cfg, ev.op, gen, ev.num.ins, bits_override=b)
        r = _row(cfg, ev, f"mantissa_truncate({b})", out, epsilon=b,
                 declared="fp32", gen=gen, tolerances=tolerances,
                 floor=floor, slack=slack)
        r["note"] = "声称 fp32（24 位），实际按 b 位有效跑"
        rows.append(r)
        rulers.append(r)

    # ---- 具名 bug ----
    for name in ILLEGAL_VARIANTS[ev.op]:
        out = kernel_output(cfg, ev.op, gen, ev.num.ins, variant=name)
        # ⚠️ 全部具名变体**都声称 fp32** —— 包括 `no_quantize_declared_fp32`。
        #    第一版给它填了 `declared="bf16"`，于是它的门 2 头寸是
        #    `eff - (8 - bias)` ≈ 0 ⇒ **门 2 放行它**，而它正是「声明与实现不符」
        #    的样本。那个错误会让头条对照表印出「门 2 也分不开」——
        #    与结论相反，且完全看不出来。
        declared = "fp32"
        r = _row(cfg, ev, name, out, epsilon=None, declared=declared, gen=gen,
                 tolerances=tolerances, floor=floor, slack=slack)
        r["note"] = {
            "drop_tail_rows": "分块没处理 M 方向的尾巴 ⇒ 整块输出留在 0",
            "drop_tail": "分块没处理 N 方向的尾巴 ⇒ 尾部列留在 0",
            "skip_normalization": "softmax 少了除以分母那一步",
            "no_centering": "layernorm 少了减均值那一步",
            "eps_zero": "layernorm 把 eps 偷偷设成 0",
            "no_quantize_declared_fp32": "**声称 fp32，实际按 bf16 的 8 位跑**",
        }[name]
        rows.append(r)

    # ---- 逐级：门 1 单独放行了几个非法 kernel ----
    by_tol: list[dict[str, Any]] = []
    for t in tolerances:
        g1 = [r["variant"] for r in rows if r["accepted_at"][str(t)]]
        g2 = [r["variant"] for r in rows
              if r["accepted_at"][str(t)] and r["gate2_bits_ok"]]
        by_tol.append({
            "tolerance": t,
            "n_accepted_by_gate1": len(g1),
            "n_accepted_by_both_gates": len(g2),
            "accepted_by_gate1": g1,
            "accepted_by_both_gates": g2,
            "gate1_only": sorted(set(g1) - set(g2)),
        })

    # ---- 安全窗口：全部合法通过 ∧ 全部非法被拒 ----
    win = safe_window(cfg, ev, tolerances, rows)

    # ---- 断言：尺子网格必须横跨「合法实现能达到的最差精度」那条线 ----
    #
    # ⚠️ 第一版断言的是「至少一档在某级容差下被接受、至少一档不被接受」。
    #    那是**图省事**：阶梯顶端一直放到 1.0，于是全部 8 档都会被接受，
    #    这条断言必然触发 —— 而它触发的原因是**阶梯太长**，不是崖没被测量到。
    #
    # 真正要测的是这个：尺子上必须**同时**有
    #   - 漂移 ≤ 合法最差精度 的档（它与「诚实的低精度 kernel」**无法区分**），
    #   - 漂移 >  合法最差精度 的档（它明显比任何合法实现都差）。
    # 少任何一侧，「重叠区在哪」这个数就没被定位到 —— 而表看起来完全正常。
    n_r = len(rulers)
    legal_max = float(win["legal_max_drift"])
    below = [r["variant"] for r in rulers if r["drift"] <= legal_max]
    above = [r["variant"] for r in rulers if r["drift"] > legal_max]
    if rulers and (not below or not above):
        raise RuntimeError(
            f"尺子网格全在「合法最差精度」({legal_max:.3e}) 的同一侧："
            f"下方 {len(below)} 档 / 上方 {len(above)} 档。"
            "「合法低精度」与「偷偷降精度」在误差上重叠的那一段没有被测量到 —— "
            "请调整 adversarial.mantissa_bits。")
    # ⚠️ 这一条**必须是算出来的**：写死成 `True` 的话，
    #    「网格横跨了那条线」在产物里永远成立 —— 它看起来像测量结果，其实是假设。
    spans = bool(rulers) and bool(below) and bool(above)


    default_tol = float(cfg["semantics"]["tolerance"])
    d_acc = [r["variant"] for r in rows if r["accepted_at"][str(default_tol)]]
    d_only = [r["variant"] for r in rows if r["accepted_at"][str(default_tol)]
              and not r["gate2_bits_ok"]]

    return {
        "op": ev.op,
        "variants": rows,
        "rulers": rulers,
        "n_variants": len(rows),
        "n_rulers": n_r,
        "by_tolerance": by_tol,
        "default_tolerance": default_tol,
        "n_accepted_at_default_tolerance": len(d_acc),
        "accepted_at_default_tolerance": d_acc,
        "n_gate1_only_at_default_tolerance": len(d_only),
        "gate1_only_at_default_tolerance": d_only,
        "ruler_grid_spans_cliff": spans,
        "rulers_below_legal_max": below,
        "rulers_above_legal_max": above,
        "n_rulers_below_legal_max": len(below),
        "n_rulers_above_legal_max": len(above),
        "safe_window": win,
        "n_gate1_only_anywhere": sum(1 for r in rows
                                     if r["gate1_only_would_accept"]),
        "gate1_only_anywhere": [r["variant"] for r in rows
                                if r["gate1_only_would_accept"]],
    }


def safe_window(cfg: dict[str, Any], ev: KernelEvaluator,
                tolerances: Sequence[float],
                illegal_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """**安全容差窗口**：`[全部合法 kernel 的最大漂移, 全部非法 kernel 的最小漂移)`。

    ⚠️ 这里的「合法 kernel」取**整个合法空间**的漂移上界，
    不是「基线自己」。第一版拿基线比（漂移 0），于是窗口看起来从 0 开始 ——
    而 0 是**基线自己**，不是「合法实现能达到的最差精度」。
    """
    legal_max = max(t.drift for t in ev.space_all)
    ill = [float(r["drift"]) for r in illegal_rows]
    illegal_min = min(ill) if ill else float("nan")
    nonempty = bool(ill) and illegal_min > legal_max
    return {
        "op": ev.op,
        "legal_max_drift": round(legal_max, 15),
        "illegal_min_drift": (round(illegal_min, 15) if ill else None),
        "nonempty": nonempty,
        "lo": round(legal_max, 15),
        "hi": (round(illegal_min, 15) if ill else None),
        "width_decades": (round(__import__("math").log10(illegal_min / legal_max), 6)
                          if nonempty and legal_max > 0 else None),
        "witness_legal": max(ev.space_all, key=lambda t: t.drift).genome.label(),
        "witness_illegal": (min(illegal_rows, key=lambda r: r["drift"])["variant"]
                            if ill else None),
        "note": "窗口非空 ⟺ 存在一个 τ 同时「收下全部合法精度」与「拒掉全部非法扰动」。"
                "**空了是结论，不是 bug** —— 它说明精度与 bug 在误差上重叠了。",
    }


def cross_op_headline(adv: dict[str, Any], ev: KernelEvaluator) -> dict[str, Any]:
    """把「诚实的 bf16」与「声称 fp32 实际 bf16」并排放。

    **这是本项目最想印的一个对比**：两行的漂移同量级、门 1 的结论相同，
    而门 2 分开它们。

    ## ⚠️ 「诚实的 bf16」有**两个**代表，它们回答两个不同的问题

    第一版只取「漂移最大的那个诚实 bf16」，然后断言「门 1 对两行结论相同」。
    实测在 matmul 上**红了** —— 因为最差的那个诚实 bf16 的漂移可以**大于**
    假 fp32 的漂移（3e-2 vs 5e-3），于是门 1 在两行上给出**相反**的结论。
    那不是结论错，是**代表选错了**：拿「最差的诚实 bf16」去比「某一个具体的
    假 fp32」，量到的是「这两个 kernel 差多少」，不是
    「假 fp32 能不能混进诚实 bf16 里」。

    所以这里报两个代表：

    - `honest_bf16_worst`：漂移最大的诚实 bf16 —— 它定义**上界**；
    - `nearest_honest_bf16`：漂移**最接近**假 fp32 的那个诚实 bf16 ——
      它才是「最难区分的那一对」，而门 1 / 门 2 的对照必须在**这一对**上做。

    再加上 `fake_inside_honest_range`（假 fp32 的漂移落在全部诚实 bf16 的
    漂移区间**之内**）：这一条是可被推翻的 —— 若那个变体实际按 2 位跑，
    它的漂移会远大于任何诚实 bf16，于是它落在区间之外。
    """
    honest = [t for t in ev.space_all if t.genome.accum_dtype == "bf16"]
    fake = next((r for r in adv["variants"]
                 if r["variant"] == "no_quantize_declared_fp32"), None)
    # ⚠️ 早退分支也要**把键补全**：只返回 `{"available": False}`
    #    会让调用方 `h["honest_bf16"]` 抛 KeyError —— 一个「对照表少了一节」
    #    的问题伪装成「代码崩了」，而两者该有不同的处置。
    if not honest:
        return {"op": ev.op, "available": False,
                "why": "空间里没有 bf16 的合法 kernel —— 对照组不存在"}
    if fake is None:
        return {"op": ev.op, "available": False,
                "why": "非法变体里没有 no_quantize_declared_fp32 —— 对照组不存在"}

    worst = max(honest, key=lambda t: (t.drift, t.genome.digest()))
    nearest = min(honest, key=lambda t: (abs(t.drift - fake["drift"]),
                                         t.genome.digest()))
    drifts = [t.drift for t in honest]
    lo, hi = min(drifts), max(drifts)
    fake_drift = float(fake["drift"])
    tol = 1e-2
    nearest_gate1 = bool(nearest.drift <= tol)
    fake_gate1 = bool(fake_drift <= tol)
    return {
        "op": ev.op,
        "available": True,
        "honest_bf16": {
            "label": worst.genome.label(),
            "declared_dtype": worst.genome.accum_dtype,
            "declared_bits": worst.declared_bits,
            "drift": round(worst.drift, 15),
            "effective_bits": round(worst.effective_bits, 9),
            "gate1_feasible_at_1e-2": bool(worst.drift <= tol),
            "gate2_bits_ok": bool(worst.bits_ok),
        },
        "honest_bf16_range": {
            "min_drift": round(lo, 15),
            "max_drift": round(hi, 15),
            "n": len(honest),
            "witness_min": min(honest, key=lambda t: (t.drift,
                                                      t.genome.digest())).genome.label(),
            "witness_max": worst.genome.label(),
        },
        "fake_fp32": {
            "label": "baseline 基因组，但实际按 bf16 跑",
            "declared_dtype": fake["declared_dtype"],
            "declared_bits": fake["declared_bits"],
            "drift": fake_drift,
            "effective_bits": fake["effective_bits"],
            "gate1_feasible_at_1e-2": fake_gate1,
            "gate2_bits_ok": fake["gate2_bits_ok"],
        },
        "nearest_honest_bf16": {
            "label": nearest.genome.label(),
            "declared_dtype": nearest.genome.accum_dtype,
            "declared_bits": nearest.declared_bits,
            "drift": round(nearest.drift, 15),
            "effective_bits": round(nearest.effective_bits, 9),
            "gate1_feasible_at_1e-2": nearest_gate1,
            "gate2_bits_ok": bool(nearest.bits_ok),
            "drift_ratio": (round(fake_drift / nearest.drift, 6)
                            if nearest.drift else None),
            "bit_identical_drift": bool(fake_drift == nearest.drift),
        },
        #: **门 1 分不开这一对**：两行的漂移同量级、门 1 结论相同。
        "gate1_same_verdict": bool(nearest_gate1 == fake_gate1),
        #: **门 2 分得开这一对**：诚实的那行过、假装的那行被拒。
        "gate2_separates": bool(nearest.bits_ok and not fake["gate2_bits_ok"]),
        "drift_ratio": (round(fake_drift / nearest.drift, 6)
                        if nearest.drift else None),
        "fake_inside_honest_range": bool(lo <= fake_drift <= hi),
        "note": "两行漂移同量级、门 1 结论相同；**门 2 是唯一分开它们的判据**。"
                "对照做在 `nearest_honest_bf16`（最难区分的那一对）上 —— "
                "拿「最差的诚实 bf16」去比一个具体的假 fp32，量到的是"
                "「这两个 kernel 差多少」，而不是「假 fp32 混不混得进去」。",
    }


__all__ = ["run_adversarial", "safe_window", "cross_op_headline"]
