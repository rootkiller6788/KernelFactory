"""基准：解析延迟 vs **实测求值耗时**。

## 它量的是什么

不是「kernel 有多快」——本机没有 GPU，那个数**量不出来**。
它量的是「解析成本模型给出的排序，与真的跑一遍数值孪生的耗时，同不同向」。

⚠️ **这是一个关于孪生自身一致性的读数，不是一个性能读数。**
把它的结论写成「我们验证了 kernel 的性能」是这一类报告最常见的越界。

## 时序读数必须落进**不进版本摘要**的产物

实测微秒数随机器变。`L6-graphopt` 实测抓过：`analytic_latency_tracks_wallclock`
的证据里嵌着每次求值的微秒数，而 `expectations.json` 是**进摘要**的 ⇒
「同一份输入跑两次」得到了两个版本号，版本号的意义当场失效（附录 C 286）。
所以这条期望的读数走 `benchmark.json`，而 `expectations.json` 里只留**条数**。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .evaluate import KernelEvaluator
from .genome import KernelGenome
from .ops import kernel_output


def wallclock_probe(cfg: dict[str, Any], ev: KernelEvaluator,
                    genomes: Sequence[KernelGenome],
                    analytic: dict[str, float]) -> dict[str, Any]:
    """对每个基因组**绕过缓存**重跑一遍数值，记录耗时。

    ⚠️ 必须绕过缓存：缓存命中是字典查找，与 kernel 的复杂度无关 ——
    用它当「实测耗时」会得到一个**与基因组无关的常数**，
    而那个常数与解析延迟的秩相关恰好接近 0，看起来像「模型不准」。
    """
    rows: list[dict[str, Any]] = []
    for g in genomes:
        t0 = time.perf_counter()
        kernel_output(cfg, ev.op, g, ev.num.ins)
        dt = time.perf_counter() - t0
        rows.append({
            "genome_id": g.id,
            "label": g.label(),
            "analytic_latency_us": round(analytic[g.digest()], 9),
            "measured_eval_us": round(dt * 1e6, 6),
        })
    return {
        "op": ev.op,
        "n": len(rows),
        "rows": rows,
        "note": "**这是孪生自身一致性的读数，不是性能读数** —— "
                "本机没有 GPU，「kernel 有多快」这个数量不出来。",
    }


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """秩相关（Spearman）。并列值取平均秩。

    ⚠️ 用秩相关而不是皮尔逊：解析延迟与实际耗时的关系**不假定线性**
    （一个是 µs，一个是 µs 级的 Python 循环），要问的是「排序同不同向」。
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return float("nan")

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    dx = [a - mx for a in rx]
    dy = [b - my for b in ry]
    nx = sum(a * a for a in dx) ** 0.5
    ny = sum(b * b for b in dy) ** 0.5
    if nx == 0 or ny == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (nx * ny)


__all__ = ["wallclock_probe", "spearman"]
