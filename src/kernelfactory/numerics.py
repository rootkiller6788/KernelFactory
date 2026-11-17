"""数值孪生：把「这个 kernel 用了多少位精度」变成一个**可以跑、可以测**的量。

## 三个概念必须分清（本项目最重要的三个词）

1. **声明精度**（`accum_dtype` 基因）—— kernel **声称**自己用什么精度累加。
   它是一个**字符串**，写在 genome 里。
2. **有效位宽** `effective_bits` —— **测出来的**。手法：把**同一个算法**
   （同样的分块、同样的累加顺序）换成 fp64 重跑一遍，得到 `exact`；
   与真正跑出来的结果比，偏差就暴露了「这一步实际丢了多少位」。
   ⚠️ 它**不看声明** —— 这正是它能抓到「声称 fp32、实际截到 8 位」的原因。
3. **误差** `drift` —— 与**基线 kernel** 的输出偏差。
   ⚠️ 它是**声明精度与实现 bug 的混合读数**：一个诚实的 bf16 kernel 与一个
   偷偷截断到 8 位的 fp32 kernel，`drift` 是同一个量级。
   **本项目最值钱的结论就长在这个混淆上** —— 详见 `report.py` 的头条。

## 为什么舍入要自己写而不是靠 `float32`

Python 没有 fp32 标量，`struct.pack('f', x)` 每次要过一遍字节串（慢且难读）。
更要紧的是 **tf32（11 位有效）与 bf16（8 位有效）根本没有对应的 Python 类型**。
所以统一走 `round_to_bits(x, bits)`：`frexp` 拆出尾数、按 `bits` 定点舍入、`ldexp` 装回。
它是**精确的**（每一步只做一次二进制舍入），而且是**确定性的**。

## 有效位宽的读数带有 n 依赖的偏置（这是推出来的，不是拍的）

n 步随机舍入的累加，相对误差按 √n 增长 ⇒ `bits_eff ≈ bits_declared − 0.5·log2(n)`。
所以门槛上的 slack 不是「留点余量」，而是**这个公式给出来的**。
期望 `effective_bits_has_a_sqrt_n_bias` 就是去核这条式子。
"""

from __future__ import annotations

import math
from typing import Sequence

#: 各精度类型的**有效位宽**（含隐含的前导 1 位）。
#: - fp32：1 + 23 = 24
#: - tf32：1 + 10 = 11（NVIDIA A100 起的张量核输入格式）
#: - bf16：1 + 7 = 8（bfloat16：8 位指数，7 位尾数）
SIGNIFICANT_BITS: dict[str, int] = {"fp32": 24, "tf32": 11, "bf16": 8}

#: 精度类型的**相对加速比**（相对 fp32）。**这是假设，不是测量** ——
#: 本机没有 GPU，张量核的真实加速比量不出来。见 `cost.py` 的 `COST_SOURCE`。
DTYPE_SPEEDUP: dict[str, float] = {"fp32": 1.0, "tf32": 2.0, "bf16": 4.0}

#: fp64 的「不丢位」档：`round_to_bits(x, 53)` 是恒等变换。
FULL_BITS = 53


def round_to_bits(x: float, bits: int) -> float:
    """把 `x` 舍入到 `bits` 位有效数字（二进制）。`bits >= 53` ⇒ 恒等。

    实现口径：
    - `frexp` 给出 `x = m · 2**e`，`0.5 ≤ |m| < 1`；
    - 把 `m` 放大到 `2**bits` 后取整再缩回 ⇒ **只做一次**二进制舍入，
      不会出现「先转十进制再转回来」那种二次误差；
    - `round()` 是 round-half-even（银行家舍入），确定性且无偏。
    """
    if x == 0.0 or not math.isfinite(x):
        return x
    if bits >= FULL_BITS:
        return x
    if bits < 1:
        raise ValueError(f"bits 必须 ≥ 1，收到 {bits}")
    m, e = math.frexp(x)
    scale = float(1 << bits)
    m = round(m * scale) / scale
    return math.ldexp(m, e)


def pair_sum(values: list[float], bits: int) -> float:
    """**树形（pairwise）求和**，每一步都按 `bits` 舍入。

    为什么不是顺序求和：真实 kernel 里的规约是 `tl.sum` 那种**树形**规约，
    它的误差是 O(log n) 而不是 O(n)。这个差别正是 `tile_k` 基因能影响精度的
    物理来源 —— 顺序求和会让 `tile_k` 在误差上完全消失，
    而「一个对误差没有影响的基因」与「一个忘了接线的基因」在产物里长得一样。
    """
    if not values:
        return 0.0
    cur = [round_to_bits(v, bits) for v in values]
    while len(cur) > 1:
        nxt: list[float] = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(round_to_bits(cur[i] + cur[i + 1], bits))
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def seq_sum(values: list[float], bits: int) -> float:
    """顺序累加，每一步按 `bits` 舍入（跨分块的累加走这条）。"""
    acc = 0.0
    for v in values:
        acc = round_to_bits(acc + v, bits)
    return acc


def chunked_sum(products: list[float], tile_k: int, bits: int) -> float:
    """**分块累加**：块内树形规约、块间顺序累加。两者都按 `bits` 舍入。

    `tile_k` 是块宽。块越大 ⇒ 树形规约越深 ⇒ 误差越小；块越小 ⇒
    跨块顺序累加越多 ⇒ 误差越大。于是 `tile_k` 有了一个**方向明确的精度代价**，
    而它同时还有速度收益（更大的一块 ⇒ 更少的边界处理）——
    **这就是这个项目里最干净的一个权衡轴。**
    """
    if tile_k >= len(products):
        return pair_sum(products, bits)
    parts = [pair_sum(products[i:i + tile_k], bits)
             for i in range(0, len(products), tile_k)]
    return seq_sum(parts, bits)


def scaled_max_error(got: Sequence[float], want: Sequence[float],
                     floor: float) -> float:
    """**按问题量级归一化的最大绝对偏差**：

        err = max_i |got_i − want_i| / (max_j |want_j| + floor)

    ## 为什么**不是**逐元素相对误差（本项目踩过，附录 C 307）

    第一版用的是 `max_i |got_i − want_i| / |want_i|`。它在矩阵乘法上**完全失效**：
    一个真值恰好接近 0 的输出元素（K 项正负相消的必然产物）分母趋零，
    于是它的相对误差可以到 1e-1 量级 —— 而那个数会**盖过全部真实误差**，
    成为 `max` 的返回值。

    后果是读数变成了**输入矩阵的性质**而不是 kernel 的性质：
    fp32 的「最大相对误差」被量成 1.27e-6（比 2^−24 大 20 倍），
    tf32 被量成 1.71e-1（比 2^−11 大 300 倍）。**它们看起来像两个合理的数**，
    所以这个 bug 不报错，只让「有效位宽」这个读数整体偏低 5~8 位，
    而它印在报告里毫无异样。

    真实世界的正确性检查早就绕开了这件事：`torch.allclose` 的判据是
    `|a−b| ≤ atol + rtol·|b|` —— 那个 `atol` 就是承认「真值近零的元素
    不能按相对误差比」。本函数取 `atol = floor · max|want|`，
    于是**分母永远由问题量级主导**。`floor` 只用来防「整个输出全是 0」的 `0/0`。
    """
    if len(got) != len(want):
        raise ValueError(f"长度不一致：{len(got)} vs {len(want)}")
    if not want:
        return 0.0
    scale = max((abs(w) for w in want), default=0.0)
    denom = scale + floor
    if denom <= 0.0:
        return 0.0
    return max((abs(g - w) for g, w in zip(got, want)), default=0.0) / denom


def effective_bits(err: float) -> float:
    """相对误差 → **有效位宽**。`err = 0` ⇒ `FULL_BITS`（没有丢任何位）。

    ⚠️ 这是一个**读数**，不是一个精确的整数：n 步累加的误差按 √n 增长，
    所以它系统性地低于声明位宽（见模块文档里的公式）。
    门槛上比的是「不低于声明位宽 − 偏置 − slack」，**不是**「等于声明位宽」。
    """
    if err <= 0.0:
        return float(FULL_BITS)
    return -math.log2(err)


def expected_bits_bias(n_steps: int) -> float:
    """√n 偏置的**解析预测**：`0.5 · log2(n)`。`n_steps = 1` ⇒ 0。

    ⚠️ 这个式子的前提是「每一步的舍入**独立**」。它对**顺序累加**成立
    （每一步的舍入误差与当前部分和无关），对**树形规约**只是近似
    （树的不同层之间相关）。所以期望 `effective_bits_has_a_sqrt_n_bias`
    核的是「在 1 位以内」，不是「精确相等」——
    **一个只在严格相等时才成立的经验公式，与一个错误公式没有区别。**

    它本身是**推出来的**：n 步独立舍入的方差按 n 线性增长 ⇒ 标准差按 √n。
    所以期望 `effective_bits_has_a_sqrt_n_bias` 有资格去核对它 ——
    被推翻的话，说明舍入模型与「独立舍入」这个前提不符。
    """
    if n_steps <= 1:
        return 0.0
    return 0.5 * math.log2(float(n_steps))
