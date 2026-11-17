"""解析成本模型：`Latency(K)` 与 `Throughput(K)`。

## ⚠️ 这个模块里的每一个常数都是**假设**，不是测量

本机没有可用 GPU（`nvidia-smi` 起不来），也没有 torch / triton。
所以「张量核比 fp32 快 4 倍」这句话在这里**没有被测过** ——
它是从公开资料里取的一个量级，写进 `COST_SOURCE` 并被显式标注。

**把假设标成测量是这类报告最危险的失真**（`L5-quantfactory` 的附录 C）。
所以：

1. 每个常数都在 `COST_SOURCE` 里有条目，说明它是什么、从哪来、可不可信；
2. 报告里 `baseline.json.cost_source` 与 `benchmark.json.cost_sweep` 两处都印出来；
3. 有一条**敏感度扫描** `cost_sweep`：把最可疑的那个常数（`launch_overhead_us`）
   从 0 扫到 30 µs，看结论会不会翻。**一个只在某一个常数取值下成立的结论，
   不是结论，是那个常数的性质。**

## 延迟的形状

```
latency = launch_overhead + max(compute_us, memory_us) / efficiency
```

`max` 而不是相加：计算与访存重叠，瓶颈是大的那一头。
`efficiency` 是若干因子的乘积（tile / warp / stage / layout / vec / sched），
每一项都对应一个基因 —— 于是**「哪个基因值钱」这个问题在成本模型里是可分解的**，
而它分不分得动，本身就是 `genes.json` 要报的读数。
"""

from __future__ import annotations

import math
from typing import Any

from .numerics import DTYPE_SPEEDUP

#: 每个成本常数的来源。**`kind` 只有两种值：`assumption` 与 `derived`。**
#: 出现第三个值就说明有人在偷偷把假设写成测量。
COST_SOURCE: dict[str, dict[str, str]] = {
    "launch_overhead_us": {
        "kind": "assumption",
        "value_from": "公开资料里单次 kernel launch 的常见量级（1~10 µs）",
        "confidence": "低 —— 所以它有一条专门的敏感度扫描",
    },
    "flops_per_us": {
        "kind": "assumption",
        "value_from": "把一块消费级 GPU 的 fp32 峰值按 60% 实测算下来",
        "confidence": "中 —— 只影响绝对延迟，不影响排序",
    },
    "bytes_per_us": {
        "kind": "assumption",
        "value_from": "显存带宽量级除以一个保守的有效率",
        "confidence": "中",
    },
    "ai_saturate": {
        "kind": "assumption",
        "value_from": "分块算术强度的饱和点（flops/byte）",
        "confidence": "低 —— 但它只影响不同分块之间的**相对**排序",
    },
    "dtype_bytes_per_elem": {
        "kind": "assumption",
        "value_from": "fp32/tf32 = 4 字节（tf32 在显存里就是 fp32）、bf16 = 2 字节",
        "confidence": "高 —— 这是三种格式的物理定义，不是估计",
    },
    "dtype_speedup": {
        "kind": "assumption",
        "value_from": "张量核相对 fp32 的公开加速比量级（tf32≈2×、bf16≈4×）",
        "confidence": "低 —— 本机没有 GPU，量不出来。它是本项目最大的一个假设",
    },
    "tile_eff": {
        "kind": "derived",
        "value_from": "分块的算术强度 / 机器的平衡点（flops_per_us ÷ bytes_per_us）",
        "confidence": "中 —— 算术强度是真的算出来的，饱和点是个假设",
    },
    "warp_eff": {
        "kind": "derived",
        "value_from": "并行度收益的幂律 + 高 warp 数的寄存器压力惩罚",
        "confidence": "低 —— 但它给出一个**峰值在 4 个 warp** 的形状",
    },
    "stage_eff": {
        "kind": "derived",
        "value_from": "软件流水线隐藏访存延迟，收益递减",
        "confidence": "低",
    },
    "layout_eff": {
        "kind": "assumption",
        "value_from": "naive 有 bank conflict / 非合并访问；swizzled 与 vectorized 依次更好",
        "confidence": "低",
    },
    "sched_eff": {
        "kind": "assumption",
        "value_from": "persistent > pipelined > default 的经验排序",
        "confidence": "低",
    },
}


def _tile_eff(cfg: dict[str, Any], tile_m: int, tile_n: int, tile_k: int,
              bpe: int) -> float:
    """分块的**算术强度**（flops / byte）除以饱和点，上限 1。

    算术强度是真的算出来的：`2·tm·tn·tk / ((tm·tk + tk·tn + tm·tn)·bpe)`。
    饱和点 `cost.ai_saturate` 才是假设 —— 两者分开，
    于是「结论对饱和点有多敏感」是可问的。

    ⚠️ 第一版把饱和点取成 `flops_per_us ÷ bytes_per_us`（机器平衡点），
    那让 `tile_eff` 落在 0.02~0.08 —— 也就是「延迟比 roofline 大 12~50 倍」。
    真实的 512³ matmul 不会慢 50 倍。**机器平衡点量的是「问题算不算访存瓶颈」，
    分块效率量的是「这个分块能榨出多少并行度」—— 两个不同的东西，
    用同一个数当分母会让其中一个失去意义。**
    """
    c = cfg["cost"]
    tile_bytes = (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) * bpe
    if tile_bytes <= 0:
        return 0.0
    ai = 2.0 * tile_m * tile_n * tile_k / tile_bytes
    sat = float(c["ai_saturate"])
    return min(1.0, ai / sat) if sat > 0 else 1.0


def _warp_eff(num_warps: int) -> float:
    """并行度收益（幂律，指数 0.35）+ 高 warp 数的寄存器压力惩罚（每个 −2%）。"""
    base = min(1.0, (num_warps / 4.0) ** 0.35)
    return base * (1.0 - 0.02 * max(0, num_warps - 4))


def efficiency_factors(cfg: dict[str, Any], gen: Any, op: str) -> dict[str, float]:
    """把总效率**分解**成因子 —— 报告要印这张表，不然「效率」是个黑箱数。"""
    c = cfg["cost"]
    row = bool(cfg["ops"][op]["row_wise"])
    bpe = int(c["dtype_bytes_per_elem"][gen.accum_dtype])
    t_m = 1 if row else gen.tile_m
    t_k = 1 if row else gen.tile_k
    return {
        "tile_eff": (_tile_eff(cfg, t_m, gen.tile_n, t_k, bpe) if not row
                     else _tile_eff(cfg, 1, gen.tile_n, 1, bpe)),
        "warp_eff": _warp_eff(gen.num_warps),
        "stage_eff": 1.0 + 0.06 * (gen.num_stages - 1),
        "layout_eff": float(c["layout_eff"][gen.memory_layout]),
        "vec_eff": float(c["vec_eff"][str(gen.vector_width)]),
        "sched_eff": float(c["sched_eff"][gen.scheduling_strategy]),
    }


def metrics(cfg: dict[str, Any], op: str, gen: Any) -> dict[str, Any]:
    """一个基因组的**成本读数**。全部是解析值（不是测量）。

    ## 精度轴必须同时进「算力」与「访存」两侧

    第一版只让 `accum_dtype` 影响算力（`DTYPE_SPEEDUP`），访存侧一律按 4 字节算。
    后果是**行式算子（softmax / layernorm）上精度轴完全死了**：
    它们的 roofline 是访存，而访存与精度无关 ⇒ 三个精度档的延迟**一模一样**，
    阶梯从下到上一条平线。

    而那个平线看起来像一条**结论**（「放宽容差买不到速度」），
    其实是一个**建模遗漏**。所以：

    - 低精度输入的字节数真的更小（bf16 是 fp32 的一半）——
      `dtype_bytes_per_elem` 把这件事接进访存侧；
    - `tf32` 仍是 4 字节（它在显存里就是 fp32，只有乘法器截断），
      所以 **tf32 只买算力、不买带宽** —— 这是它与 bf16 的物理差别，
      不是模型里的一个巧合。

    ⇒ 于是「bf16 在访存瓶颈的算子上也能赢」这件事**是被建模出来的**，
    而不是被假设的；而它在 512³ 这种计算瓶颈的 matmul 上赢多少，
    取决于 roofline 的哪一侧占上风。**两条结论都能被实测数据核对。**
    """
    c = cfg["cost"]
    m, n, k = (int(x) for x in cfg["ops"][op]["production"])
    row = bool(cfg["ops"][op]["row_wise"])
    bpe = int(c["dtype_bytes_per_elem"][gen.accum_dtype])

    if row:
        flops = float(m * n) * float(c["row_op_flops_per_elem"])
        # 行式算子要读一遍写一遍（输出仍是 fp32），另外读 gamma/beta（layernorm）
        in_bytes = float(m * n * bpe)
        out_bytes = float(m * n * int(c["dtype_bytes_per_elem"]["fp32"]))
        extra = float(n * 2 * int(c["dtype_bytes_per_elem"]["fp32"])
                      if op == "layernorm" else 0)
        bytes_ = in_bytes + out_bytes + extra
        tiled = (1, gen.tile_n, 1)
    else:
        flops = 2.0 * m * n * k
        bytes_ = float((m * k + k * n) * bpe
                       + m * n * int(c["dtype_bytes_per_elem"]["fp32"]))
        tiled = (gen.tile_m, gen.tile_n, gen.tile_k)

    peak = float(c["flops_per_us"]) * DTYPE_SPEEDUP[gen.accum_dtype]
    compute_us = flops / peak
    memory_us = bytes_ / float(c["bytes_per_us"])
    roofline_us = max(compute_us, memory_us)

    fac = efficiency_factors(cfg, gen, op)
    eff = math.prod(fac.values())
    bottleneck = "compute" if compute_us >= memory_us else "memory"
    latency_us = float(c["launch_overhead_us"]) + roofline_us / eff

    return {
        "op": op,
        "flops": flops,
        "bytes": bytes_,
        "compute_us": compute_us,
        "memory_us": memory_us,
        "roofline_us": roofline_us,
        "bottleneck": bottleneck,
        "efficiency": eff,
        "efficiency_factors": {kk: round(vv, 9) for kk, vv in sorted(fac.items())},
        "latency_us": latency_us,
        "throughput_tflops": (flops / latency_us / 1e6) if latency_us > 0 else 0.0,
        "tile": {"m": tiled[0], "n": tiled[1], "k": tiled[2]},
    }


def cost_sweep(cfg: dict[str, Any], op: str, genomes: list[Any],
               base_metrics: dict[str, Any],
               best_metrics: dict[str, Any]) -> dict[str, Any]:
    """把最可疑的常数扫一遍，看结论翻不翻。

    扫的是 `launch_overhead_us`：它是「基线有多慢」的主要来源，
    而基线有多慢正是 `speedup` 的分母。**如果最优 kernel 只在某个
    launch 开销取值下才赢，那它赢的是那个常数，不是自己。**
    """
    rows: list[dict[str, Any]] = []
    for lo in [float(x) for x in cfg["cost_sweep"]["launch_overhead_us"]]:
        c2 = dict(cfg)
        c2["cost"] = dict(cfg["cost"])
        c2["cost"]["launch_overhead_us"] = lo
        b = metrics(c2, op, base_metrics["genome"])
        best = metrics(c2, op, best_metrics["genome"])
        rows.append({
            "launch_overhead_us": lo,
            "baseline_latency_us": round(b["latency_us"], 9),
            "best_latency_us": round(best["latency_us"], 9),
            "speedup": round(b["latency_us"] / best["latency_us"], 9),
            "best_still_wins": b["latency_us"] > best["latency_us"],
        })
    wins = [r for r in rows if r["best_still_wins"]]
    return {
        "op": op,
        "swept": "cost.launch_overhead_us",
        "rows": rows,
        "n_rows": len(rows),
        "n_wins": len(wins),
        "always_wins": len(wins) == len(rows),
        "speedup_range": [min(r["speedup"] for r in rows),
                          max(r["speedup"] for r in rows)],
        "note": "如果 best 只在部分取值下赢，那它赢的是这个常数，不是自己。",
    }


__all__ = ["COST_SOURCE", "metrics", "efficiency_factors", "cost_sweep",
           "_tile_eff", "_warp_eff"]
