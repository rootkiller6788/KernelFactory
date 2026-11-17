"""逐基因效应表：**哪个基因值钱，哪个基因是装饰。**

## 手法：反事实配对（counterfactual pairs）

对空间里每一个基因组 `g` 与每一个基因 `name`，把 `name` 换成它的另一档得到
`g'`；若 `g'` 仍在合法空间里，就得到一对 `(g, g')`。
`|latency(g') − latency(g)|` 与 `|drift(g') − drift(g)|` 就是这个基因的**直接效应**。

## ⚠️ 一条纪律：「测了发现没影响」与「根本没测到」必须分开报

一个基因可能**没有任何反事实配对**（所有替换都落到非法区）——
那时 `max_delta = 0`，而它读起来与「这个基因完全没用」一模一样。
所以每一行都必须报 `n_pairs`；`n_pairs == 0` 的行要**显式标注为「未测到」**。

这与「一条从不触发的用例与没有用例长得一样」是同一条纪律。

## 本项目的三个预期结论

1. `tile_m` / `tile_n` / `vector_width` / `memory_layout` / `num_stages`
   / `scheduling_strategy` 的 **drift 效应恰好为 0** —— 它们不改变加法树。
2. `tile_k` / `num_warps` 的 drift 效应**非零但很小**（它们是精度次项）。
3. `accum_dtype` 的 drift 效应**压倒性**（它是主项）。

第 1 条如果被推翻，说明数值孪生里有个基因偷偷改了什么 —— 那是 bug。
"""

from __future__ import annotations

from typing import Any, Sequence

from .evaluate import Trial
from .genome import GENE_ORDER


def gene_effects(trials: Sequence[Trial], op: str) -> dict[str, Any]:
    by_digest = {t.genome.digest(): t for t in trials}
    by_key: dict[tuple[Any, ...], Trial] = {
        tuple(getattr(t.genome, n) for n in GENE_ORDER): t for t in trials}

    rows: list[dict[str, Any]] = []
    for name in GENE_ORDER:
        i = GENE_ORDER.index(name)
        vals = sorted({getattr(t.genome, name) for t in trials},
                      key=lambda v: (str(type(v)), v))
        lat_delta: list[float] = []
        drift_delta: list[float] = []
        # 每个取值的**边际均值**（ANOVA 式）：它比「最大配对差」稳，
        # 因为配对差会被单个极端点主导。
        lat_mean: dict[str, list[float]] = {}
        drift_mean: dict[str, list[float]] = {}
        for t in trials:
            k = str(getattr(t.genome, name))
            lat_mean.setdefault(k, []).append(t.latency_us)
            drift_mean.setdefault(k, []).append(t.drift)

        for t in trials:
            base = list(getattr(t.genome, n) for n in GENE_ORDER)
            for v in vals:
                if v == base[i]:
                    continue
                cand = list(base)
                cand[i] = v
                o = by_key.get(tuple(cand))
                if o is None:
                    continue
                lat_delta.append(abs(o.latency_us - t.latency_us))
                drift_delta.append(abs(o.drift - t.drift))

        def spread(d: dict[str, list[float]]) -> float:
            ms = [sum(v) / len(v) for v in d.values() if v]
            return (max(ms) - min(ms)) if ms else 0.0

        rows.append({
            "op": op,
            "gene": name,
            "n_values": len(vals),
            "values": [str(v) for v in vals],
            "n_pairs": len(lat_delta),
            "max_latency_delta_us": round(max(lat_delta), 9) if lat_delta else 0.0,
            "max_drift_delta": round(max(drift_delta), 15) if drift_delta else 0.0,
            "marginal_latency_spread_us": round(spread(lat_mean), 9),
            "marginal_drift_spread": round(spread(drift_mean), 15),
            "numerically_inert": (len(drift_delta) > 0
                                  and max(drift_delta) == 0.0),
            "measured": len(lat_delta) > 0,
            "note": ("" if lat_delta else
                     "**没有任何反事实配对** —— 这不是「没影响」，是「没测到」"),
        })

    inert = [r["gene"] for r in rows if r["numerically_inert"]]
    active = [r["gene"] for r in rows if r["measured"] and not r["numerically_inert"]]
    unmeasured = [r["gene"] for r in rows if not r["measured"]]
    return {
        "op": op,
        "rows": rows,
        "numerically_inert_genes": inert,
        "numerically_active_genes": active,
        "unmeasured_genes": unmeasured,
        "n_inert": len(inert),
        "n_active": len(active),
        "note": "`numerically_inert` = 全部反事实配对的 drift 差**恰好为 0**。"
                "「恰好」是关键：一个 1e-18 的差会让它变成 active，"
                "而 1e-18 在 24 位精度下是不可表示的。",
    }


def gene_coverage(trials: Sequence[Trial]) -> dict[str, Any]:
    """每个基因在空间里的**取值个数**（=1 的基因不参与搜索）。"""
    out: dict[str, int] = {}
    for name in GENE_ORDER:
        out[name] = len({getattr(t.genome, name) for t in trials})
    return {
        "cardinality": out,
        "non_searching": sorted(n for n, c in out.items() if c <= 1),
        "note": "=1 的基因不参与搜索 —— 「不参与搜索」与「忘了接线」"
                "在产物里长得一样，所以要显式列出来。",
    }


__all__ = ["gene_effects", "gene_coverage"]
