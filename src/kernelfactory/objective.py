"""三个目标轴 —— 以及**它们其实只有两个**这件事。

README 原稿在流水线图里写了三条读数：`Latency / Throughput / Accuracy`。
本模块把它们落成三个目标：

| 轴 | 定义 | 口径 |
|---|---|---|
| `speedup` | `基线延迟 ÷ 本 kernel 延迟` | 越大越好 |
| `throughput` | `flops ÷ 延迟` | 越大越好 |
| `effective_bits` | **测出来的**有效位宽（见 `numerics`） | 越大越好 |

⚠️ **前两轴在数学上完全共线**：对固定算子，`flops` 与基线延迟都是常数 ⇒
`throughput ≡ speedup × (flops / 基线延迟)`，相关系数**恰好 1.0**。

这不是 bug，是 README 那三条读数里的一个真实冗余。**必须报出来**
（`expectations.three_objectives_are_independent` 就是去核它、并被推翻的那条）：
把「两个数是一回事」当成「两个目标」会让人以为前沿是三维的，
而三维前沿比二维前沿**看起来更有说服力**。

第三轴 `effective_bits` 与第一轴的关系是**真权衡**：
`bf16` 快 4 倍但只有 8 位有效 ⇒ 两轴负相关。
**于是「三维实际是二维」这个结论必须写成「二维独立轴 + 一维冗余」，
不能写成「一维」** —— 说成一维就把那个真权衡也一起抹掉了。
"""

from __future__ import annotations

from typing import Any

OBJECTIVES: tuple[str, ...] = ("speedup", "throughput", "effective_bits")

#: 各轴的**归一化标尺**（全空间最大值，只在空间层面算一次）。
#: 它必须与容差 / 策略 / 搜索器无关 —— 否则「放宽容差」会同时移动标尺。
OBJECTIVE_SCALE_KEYS: dict[str, str] = {
    "speedup": "max_speedup",
    "throughput": "max_throughput",
    "effective_bits": "max_effective_bits",
}


def vector(speedup: float, throughput: float, effective_bits: float
           ) -> tuple[float, float, float]:
    return (float(speedup), float(throughput), float(effective_bits))


def normalized(vec: tuple[float, float, float], scale: dict[str, float]
               ) -> tuple[float, float, float]:
    """按全空间最大值归一化。**标尺为 0 时返回 0**（不是除零崩）。"""
    return tuple((vec[i] / scale[OBJECTIVES[i]]) if scale[OBJECTIVES[i]] else 0.0
                 for i in range(len(OBJECTIVES)))        # type: ignore[return-value]


def tchebycheff(vec: tuple[float, float, float], scale: dict[str, float],
                weights: tuple[float, ...] | None = None) -> float:
    """等权 Tchebycheff 标量化（最大化口径 ⇒ 取 min 后加权）。

    为什么是 Tchebycheff 而不是加权和：加权和会把「某一维极差但其余维极好」
    的候选排到前面，而这里最差的那一维恰恰是**有效位宽** ——
    一个速度极快但只有 2 位有效的 kernel 不该因为速度拿满分。
    """
    w = weights or tuple(1.0 / len(OBJECTIVES) for _ in OBJECTIVES)
    nv = normalized(vec, scale)
    return min(w[i] * nv[i] for i in range(len(OBJECTIVES)))


def is_independent(corr: dict[str, dict[str, float]], threshold: float) -> bool:
    """两两 |corr| < threshold。`nan`（常数轴）**算作不独立** ——
    一个常数轴不是「与别的轴无关」，是「它不携带信息」。"""
    for i, a in enumerate(OBJECTIVES):
        for b in OBJECTIVES[i + 1:]:
            c = corr.get(a, {}).get(b)
            if c is None or c != c or abs(c) >= threshold:
                return False
    return True


def as_dict() -> dict[str, Any]:
    return {
        "objectives": list(OBJECTIVES),
        "direction": "全部最大化",
        "redundancy_note": "speedup 与 throughput 对固定算子**完全共线**"
                           "（相关系数恰好 1.0）—— 三条读数里有一条是重复的。",
        "scale_keys": dict(OBJECTIVE_SCALE_KEYS),
    }


__all__ = ["OBJECTIVES", "OBJECTIVE_SCALE_KEYS", "vector", "normalized",
           "tchebycheff", "is_independent", "as_dict"]
