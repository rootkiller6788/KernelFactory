"""KernelGenome：9 个基因 + 候选空间。

## ⚠️ 为什么是 9 个基因，而 README 原稿写的是 8 个

README 的设计稿列了 8 个基因：`tile_m / tile_n / tile_k / num_warps /
num_stages / memory_layout / vector_width / scheduling_strategy`。
**一个精度轴都没有。**

后果是致命的：本项目的主命题是 `K* = arg min_K Latency(K) subject to Error(K) < ε`，
而**如果精度不在基因里，`Error(K)` 在基因空间上就是常数** ——
那条硬约束要么永远成立、要么永远不成立，它不构成一个搜索问题。
「一个在空间上恒定的约束」与「没有约束」在产物里长得一模一样。

所以加第 9 个基因 `accum_dtype ∈ {fp32, tf32, bf16}`。它不是装饰：
- 它是 `Error(K)` 的**主项**（输入量化误差 ≈ 2^−bits）；
- 它是 `Latency(K)` 的**主项之一**（张量核加速比 1× / 2× / 4×，**这是假设**）；
- 它正是 README 那句「没有它，搜索会收敛到算得快但算错了的 kernel」所指的那个轴。

⇒ 8 → 9 是一次**被理由驱动的修订**，不是随手加维。原稿的 8 个基因一个没少。

## 三个「不参与搜索的基因」要显式报出来

行式算子（softmax / layernorm）沿 N 维规约，`tile_m` 与 `tile_k` 恒为 1
（`tile_k` 在行式算子上没有「K 维」可切）。于是这两个基因在行式算子上
**取值个数 = 1**，对搜索没有任何贡献。

**「一个不参与搜索的基因」与「一个忘了接线的基因」在产物里长得一样** ——
所以 `genes.json` 必须逐基因报出**取值个数**与**去重后的取值个数**，
而不是只报「一共 9 个基因」。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Iterator, Sequence

#: 基因名（顺序即 `to_dict` 的字段顺序，也是报告里的列顺序）。
GENE_ORDER: tuple[str, ...] = (
    "tile_m", "tile_n", "tile_k",
    "num_warps", "num_stages",
    "memory_layout", "vector_width", "scheduling_strategy",
    "accum_dtype",
)

#: 行式算子上恒为 1 的基因（它们**不参与搜索**）。
ROW_CONSTANT_GENES: tuple[str, ...] = ("tile_m", "tile_k")


@dataclass(frozen=True)
class KernelGenome:
    """一个 kernel 配置。**不可变 ⇒ 可哈希 ⇒ 可直接做缓存键与去重键。**

    ⚠️ 别用 `id()` 或 `hash()`（内置 hash 受 `PYTHONHASHSEED` 影响，
    跨进程不稳定）；去重一律走 `digest()`。
    """

    tile_m: int
    tile_n: int
    tile_k: int
    num_warps: int
    num_stages: int
    memory_layout: str
    vector_width: int
    scheduling_strategy: str
    accum_dtype: str

    # ---------------------------------------------------------------- 身份
    def __post_init__(self) -> None:
        """**指纹在构造时算一次**。

        ⚠️ 这不是微优化。搜索器与邻域表要反复问同一个基因组的指纹：
        一次 `run` 里的调用次数是 **百万级**，每次重算一遍 sha256 会让
        整条流水线从 3 秒变成跑不完。缓存到实例上（`object.__setattr__`
        绕过 frozen）是安全的 —— 基因组不可变，指纹是它的纯函数。
        """
        raw = "|".join(f"{n}={getattr(self, n)!r}" for n in GENE_ORDER)
        object.__setattr__(
            self, "_digest",
            hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12])

    def digest(self) -> str:
        """稳定短指纹。**与字段顺序无关**（按名字拼），改字段名才会变。"""
        return self._digest

    @property
    def id(self) -> str:
        return "g:" + self.digest()

    def label(self) -> str:
        return (f"{self.accum_dtype}/T{self.tile_m}x{self.tile_n}x{self.tile_k}"
                f"/w{self.num_warps}/s{self.num_stages}"
                f"/{self.memory_layout[:3]}/v{self.vector_width}"
                f"/{self.scheduling_strategy[:4]}")

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in GENE_ORDER}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KernelGenome":
        """**显式按字段取值，绝不 `(**d)`** —— 多一个展示字段就会炸（坑 42）。"""
        return cls(**{n: d[n] for n in GENE_ORDER})

    def value_of(self, gene: str) -> Any:
        return getattr(self, gene)


# --------------------------------------------------------------------------- #
# 空间
# --------------------------------------------------------------------------- #

def gene_values(cfg: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """从配置读出每个基因的取值表（**排序后**，保证枚举顺序确定）。"""
    g = cfg["genome"]
    out: dict[str, tuple[Any, ...]] = {}
    for name in GENE_ORDER:
        vals = g[name]
        if not vals:
            raise ValueError(f"基因 {name} 的取值表为空 —— 那个基因会从空间里消失")
        # 字符串保持配置顺序（它们不是数值，排序反而会打乱「快 → 慢」的声明顺序）
        out[name] = tuple(vals) if isinstance(vals[0], str) else tuple(sorted(vals))
    return out


def is_valid(gen: KernelGenome, cfg: dict[str, Any], op: str) -> tuple[bool, str]:
    """基因合法性：**返回理由**，不返回裸布尔。

    合法性规则不是审美，每一条都对应一个真实的失败模式：
    - `tile_* <= 探测形状`：比整个矩阵还大的分块 ⇒ 分块逻辑退化成单块，
      那条基因在误差与成本上都变成常数（**看起来在搜索，其实没搜**）；
    - `vector_width <= tile_n`：向量宽度超过一行 ⇒ 越界读，实际 kernel 会崩；
    - `memory_layout == "vectorized" ⇒ vector_width >= 2`：
      叫 vectorized 却按标量走，是**声明与实现不符** ——
      与 `accum_dtype` 声称 fp32 实际截断是同一类错误；
    - `num_stages > 1 ⇒ scheduling == "pipelined"`：
      软件流水线（多 stage 预取）只有 pipelined 调度会发射它，
      其余调度下 `num_stages` 是个**没人读的旋钮**；
    - `num_warps * 32 <= 4 * tile_m * tile_n`：线程数超过工作项的 4 倍 ⇒
      多数线程空转，那不是「更快的配置」而是「没意义的配置」。
    """
    probe = op_probe_shape(cfg, op)
    m, n, k = probe
    t_m, t_n, t_k = gen.tile_m, gen.tile_n, gen.tile_k
    if t_m > m:
        return False, "tile_m_over_shape"
    if t_n > n:
        return False, "tile_n_over_shape"
    if t_k > k:
        return False, "tile_k_over_shape"
    if gen.vector_width > t_n:
        return False, "vector_width_over_tile_n"
    if gen.memory_layout == "vectorized" and gen.vector_width < 2:
        return False, "vectorized_without_vector"
    if gen.num_stages > 1 and gen.scheduling_strategy != "pipelined":
        return False, "stages_without_pipelining"
    if gen.num_warps * 32 > 4 * t_m * t_n:
        return False, "too_many_warps_for_tile"
    return True, "ok"


def op_probe_shape(cfg: dict[str, Any], op: str) -> tuple[int, int, int]:
    """算子用于**数值孪生**的探测形状（小，但要够深）。"""
    return tuple(cfg["ops"][op]["probe"])          # type: ignore[return-value]


def op_production_shape(cfg: dict[str, Any], op: str) -> tuple[int, int, int]:
    """算子用于**成本模型**的产出形状（大，量级真实）。"""
    return tuple(cfg["ops"][op]["production"])     # type: ignore[return-value]


def _row_op(cfg: dict[str, Any], op: str) -> bool:
    return bool(cfg["ops"][op]["row_wise"])


def iter_raw(cfg: dict[str, Any], op: str) -> Iterator[KernelGenome]:
    """笛卡尔积展开（**未过滤**）。顺序确定：按 `GENE_ORDER` 逐位。"""
    vals = gene_values(cfg)
    row = _row_op(cfg, op)

    def rec(i: int, acc: dict[str, Any]) -> Iterator[KernelGenome]:
        if i == len(GENE_ORDER):
            yield KernelGenome(**acc)
            return
        name = GENE_ORDER[i]
        choices = vals[name]
        if row and name in ROW_CONSTANT_GENES:
            choices = (1,)          # 行式算子上这两个基因恒为 1
        for v in choices:
            acc[name] = v
            yield from rec(i + 1, acc)

    yield from rec(0, {})


def enumerate_space(cfg: dict[str, Any], op: str
                    ) -> tuple[list[KernelGenome], dict[str, Any]]:
    """展开 + 过滤，返回 `(合法基因组, 空间诊断)`。

    诊断里必须带**被过滤掉的原因分布** —— 一个「空间比预期小」的读数，
    与一个「过滤规则写错、砍掉了半个空间」的读数，在只看大小时无法区分。
    """
    raw = list(iter_raw(cfg, op))
    kept: list[KernelGenome] = []
    reasons: dict[str, int] = {}
    for g in raw:
        ok, why = is_valid(g, cfg, op)
        if ok:
            kept.append(g)
        else:
            reasons[why] = reasons.get(why, 0) + 1

    # 去重（笛卡尔积理论上不产生重复，但 ROW_CONSTANT_GENES 会把
    # tile_m/tile_k 的多档压成同一档 —— **重复会让「空间大小」虚高**）
    seen: dict[str, KernelGenome] = {}
    for g in kept:
        seen.setdefault(g.digest(), g)
    dedup = [seen[k] for k in sorted(seen)]

    diag = {
        "op": op,
        "n_raw": len(raw),
        "n_valid": len(kept),
        "n_distinct": len(dedup),
        "n_pruned": len(raw) - len(kept),
        "n_collapsed_duplicates": len(kept) - len(dedup),
        "prune_reasons": {k: reasons[k] for k in sorted(reasons)},
        "gene_cardinality": {n: len(set(getattr(g, n) for g in dedup))
                             for n in GENE_ORDER},
        "row_wise": _row_op(cfg, op),
        "note": "gene_cardinality = 该基因在**合法空间**里的取值个数。"
                "=1 的基因不参与搜索 —— 「不参与搜索」与「忘了接线」在产物里长得一样。",
    }
    return dedup, diag


def space_digest(genomes: Sequence[KernelGenome]) -> str:
    """空间指纹：**对 digest 排序后**再拼，与枚举顺序无关。"""
    h = hashlib.sha256()
    for d in sorted(g.digest() for g in genomes):
        h.update(d.encode("utf-8"))
    return h.hexdigest()[:16]


def _field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(KernelGenome))


__all__ = [
    "GENE_ORDER", "ROW_CONSTANT_GENES", "KernelGenome", "gene_values",
    "is_valid", "iter_raw", "enumerate_space", "space_digest",
    "op_probe_shape", "op_production_shape", "_field_names",
]
