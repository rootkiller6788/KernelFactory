"""支配、前沿、超体积、相关矩阵。全部**最大化**口径。

## 一个口径纪律

`dominates(a, b)` 里 `a` 支配 `b` ⟺ `a` 每一维都不差、且至少一维更好。
写成「不差 + 至少一维严格更好」时，**两个完全相同的点互不支配** ——
这是对的（重复点不该把对方踢出前沿），但它有一个后果：
「复制一个点到列表里」不会让前沿变小，**却会让 `n_frontier` 变大**。
所以 `n_frontier` 与 `n_frontier_distinct` 必须**两个都报**（附录 C 285）。

## 超体积的参考点

参考点必须**全空间唯一**（只由空间决定），不能按容差档各自取 ——
按档取会让「放宽容差」同时移动参考点，超体积的变化就分不清来源
（`L6-graphopt` 实测凭空涨 68%，全是假象，附录 C 283）。
"""

from __future__ import annotations

from typing import Any, Sequence


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        raise ValueError(f"维度不一致：{len(a)} vs {len(b)}")
    ge = all(x >= y for x, y in zip(a, b))
    gt = any(x > y for x, y in zip(a, b))
    return ge and gt


def non_dominated_indices(vecs: Sequence[Sequence[float]]) -> list[int]:
    """返回**下标**（不是向量）—— 向量可能重复，下标不会。

    ## 为什么不是朴素的双重循环

    朴素版是 O(n²)：`for i: if not any(dominates(b, a) for b in vecs)`。
    空间是 3600 个基因组、每个格子算一次前沿、一次 `run` 有上百个格子 ⇒
    13M × 100 = 十亿次比较，跑不完。

    这里用**排序扫描**：按第一维降序（并列时按后两维降序）走一遍，
    每个点只与**当前前沿**比。
    - 正确性：若 `j` 支配 `i`，则 `j` 在第一维上不劣于 `i`；
      若相等则必在第二维严格更好 ⇒ 排序把 `j` 排在 `i` 之前。
      于是 `i` 被扫到时，支配它的那个点（或一个支配它的前沿点，由传递性）
      已经在 `front` 里。
    - 复杂度 O(n·|front|)，而 `|front|` 通常只有几十。
    """
    if not vecs:
        return []
    dim = len(vecs[0])
    order = sorted(range(len(vecs)),
                   key=lambda i: tuple(-float(vecs[i][d]) for d in range(dim)))
    front: list[int] = []
    for i in order:
        a = vecs[i]
        if not any(dominates(vecs[j], a) for j in front):
            front.append(i)
    return front


def distinct_count(vecs: Sequence[Sequence[float]], ndigits: int = 12) -> int:
    """**去重后的**点数。重复点互不支配 ⇒ 会灌水 `n_frontier`。"""
    return len({tuple(round(float(x), ndigits) for x in v) for v in vecs})


def hypervolume_2d(points: Sequence[Sequence[float]],
                   reference: Sequence[float]) -> float:
    """二维超体积（被参考点支配的那块面积）。本项目只用二维投影。

    为什么不用通用 n 维超体积：n 维的实现要么慢要么错，而本项目的结论
    全部落在二维投影上（第三个目标轴与第一个**完全共线**，见
    `expectations.three_objectives_are_independent`）。**用一个只在二维
    成立的算法去算三维数，比不算更危险。**
    """
    if len(reference) != 2:
        raise ValueError("hypervolume_2d 只吃二维参考点")
    pts = sorted(((float(a), float(b)) for a, b in points),
                 key=lambda p: (-p[0], -p[1]))
    hv = 0.0
    best_y = float(reference[1])
    for x, y in pts:
        if x <= reference[0] or y <= reference[1]:
            continue
        if y > best_y:
            hv += (x - reference[0]) * (y - best_y)
            best_y = y
    return hv


def correlation_matrix(rows: Sequence[Sequence[float]],
                       names: Sequence[str]) -> dict[str, dict[str, float]]:
    """皮尔逊相关系数矩阵。样本 < 3 时返回空（算不出来就不算）。"""
    n = len(rows)
    if n < 3:
        return {}
    dim = len(names)
    cols = [[float(r[i]) for r in rows] for i in range(dim)]

    def corr(a: list[float], b: list[float]) -> float:
        ma = sum(a) / len(a)
        mb = sum(b) / len(b)
        da = [x - ma for x in a]
        db = [x - mb for x in b]
        na = sum(x * x for x in da) ** 0.5
        nb = sum(x * x for x in db) ** 0.5
        if na == 0.0 or nb == 0.0:
            return float("nan")     # 常数轴与任何轴的相关系数**无定义**，不是 0
        return sum(x * y for x, y in zip(da, db)) / (na * nb)

    return {names[i]: {names[j]: corr(cols[i], cols[j]) for j in range(dim)}
            for i in range(dim)}


def projection(points: Sequence[Sequence[float]], i: int, j: int
               ) -> list[list[float]]:
    return [[float(p[i]), float(p[j])] for p in points]


def reference_point(points: Sequence[Sequence[float]], pad: float
                    ) -> list[float]:
    """`min(各维) − pad`（最大化口径下「比最差的还差一点」）。"""
    if not points:
        raise ValueError("空集合没有参考点")
    dim = len(points[0])
    return [min(float(p[d]) for p in points) - pad for d in range(dim)]


__all__ = ["dominates", "non_dominated_indices", "distinct_count",
           "hypervolume_2d", "correlation_matrix", "projection",
           "reference_point"]
