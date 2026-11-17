"""评估一个 KernelGenome：真的跑数值、真的算成本、真的过两道门。

## 本项目的核心机制：**两层门**

README 的命题是 `K* = arg min_K Latency(K) subject to Error(K) < ε`。
这句话里只有一个门槛。本模块要说明**一个门槛不够**，并且给出第二道。

### 门 1｜一致性门（README 的那道）

`drift(K) = 内核输出与基线 kernel 输出的最大相对偏差 ≤ τ`

### 门 2｜声明门（本项目加的）

`effective_bits(K) ≥ declared_bits(K) − √n 偏置 − slack`

### 为什么必须有两道 —— 一个具体的数

设 τ = 1e-2（足够宽松，能把 bf16 放进来）。此时：

| kernel | 声明 | 实测有效位 | drift | 门 1 | 门 2 |
|---|---|---|---|---|---|
| 诚实的 bf16 | bf16 (8) | ≈ 8 − 偏置 | ~4e-3 | 过 | 过 |
| 声称 fp32、实际截到 8 位 | fp32 (24) | ≈ 8 − 偏置 | ~4e-3 | **过** | **拒** |

两行的 `drift` **同量级**、门 1 的结论**相同**。它们的区别不在误差里，
在**声明与实现是否一致**里 —— 而 `Error < ε` 这条式子根本看不到「声明」。

⇒ **「误差小」与「这个 kernel 诚实」是两件事。**
一个只报误差的验收表会把两者混为一谈，而它看起来完全正常。

### 所以交付判据也是两条

`deliverable = 门 1 ∧ 门 2`。报告的 `best_answer` 用的是这个合取，
而 `ladder.json` 里每一级**同时**报「只过门 1 的最优」与「过两道门的最优」——
两者的差距就是**单门槛设计的代价**，它是一个可以印出来的数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .cost import metrics as cost_metrics
from .genome import (
    GENE_ORDER, KernelGenome, op_probe_shape, op_production_shape,
)
from .numerics import SIGNIFICANT_BITS, effective_bits, expected_bits_bias
from .objective import tchebycheff, vector
from .ops import NumericsCache, drift_of, mode_of
from .rng import digest_of


# --------------------------------------------------------------------------- #
# 基线 kernel
# --------------------------------------------------------------------------- #

def baseline_genome(cfg: dict[str, Any], op: str) -> KernelGenome:
    """**手写基线**：`baseline.json` 里那一个具体的 kernel。

    它刻意取空间里最保守的一档（naive / 标量 / 1 个 warp / 单 stage / fp32），
    于是有充足的优化余量 —— 一个本来就接近最优的基线会让 `speedup` 失去意义。

    ⚠️ 它是**所有 drift 的比较基准**：`drift(K) = ‖K − 基线‖`。
    所以基线自己的 drift **恰好是 0**，而「与基线逐位相同」是一个
    有意义的类别（同 tile_k / 同 warp 数 / 同 dtype 的 kernel 都能落进去）。
    """
    d = cfg["baseline"][op]
    if bool(cfg["ops"][op]["row_wise"]):
        d = dict(d, tile_m=1, tile_k=1)
    return KernelGenome(**{k: d[k] for k in KernelGenome.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# 试验
# --------------------------------------------------------------------------- #

@dataclass
class Trial:
    """一次评估。**可行性、诚实性、评分是三件事**，不许合并。"""

    genome: KernelGenome
    op: str
    searcher: str
    index: int
    drift: float                 # 与**基线**的偏差 → 门 1
    error_vs_reference: float    # 与 **fp64 参考**的偏差（绝对精度）
    effective_bits: float        # 与「同算法 fp64」的偏差反解出来的位宽
    declared_bits: int
    bits_bias: float
    bits_headroom: float         # effective − (declared − bias)；门 2 的读数
    bits_ok: bool                # 门 2
    feasible: bool               # 门 1
    deliverable: bool            # 门 1 ∧ 门 2
    objectives: tuple[float, float, float]
    score: float
    latency_us: float
    speedup: float
    throughput_tflops: float
    reject_reason: str = ""
    rank: int = -1
    on_frontier: bool = False
    dominated_by: int = 0
    #: 成本读数的完整展开（效率因子分解等）。**不进 `to_dict` 的默认输出** ——
    #: 它是诊断，不是结论；把它塞进每个 trial 会让空间产物膨胀十倍。
    cost: dict[str, Any] = field(default_factory=dict)

    @property
    def genome_id(self) -> str:
        return self.genome.id

    def to_dict(self, with_metrics: bool = False, compact: bool = False
                ) -> dict[str, Any]:
        """序列化。

        `compact=True` 时**丢掉整个 `genome` 字典**（约 200 字节/点）——
        前沿动辄上千个点，而 `genome_id` 已经是它的身份，
        需要完整基因组时去 `space.json` 或 `baseline.json` 查。
        **不是为了省字节，是为了让「同一个基因组在前沿里被印了 1351 次」
        这件事不再发生。**
        """
        d = {
            "genome_id": self.genome_id,
            "label": self.genome.label(),
            "op": self.op,
            "searcher": self.searcher,
            "index": self.index,
            "drift": round(self.drift, 15),
            "error_vs_reference": round(self.error_vs_reference, 15),
            "effective_bits": round(self.effective_bits, 9),
            "declared_bits": self.declared_bits,
            "bits_bias": round(self.bits_bias, 9),
            "bits_headroom": round(self.bits_headroom, 9),
            "bits_ok": self.bits_ok,
            "feasible": self.feasible,
            "deliverable": self.deliverable,
            "objectives": [round(x, 12) for x in self.objectives],
            "score": round(self.score, 12),
            "latency_us": round(self.latency_us, 9),
            "speedup": round(self.speedup, 9),
            "throughput_tflops": round(self.throughput_tflops, 9),
            "rank": self.rank,
            "on_frontier": self.on_frontier,
            "dominated_by": self.dominated_by,
        }
        if not compact:
            d["genome"] = self.genome.to_dict()
            d["reject_reason"] = self.reject_reason
        if with_metrics:
            d["cost"] = self.cost
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        """**显式按字段取值，绝不 `(**d)`** —— 多一个展示字段就会炸（坑 42）。

        ⚠️ `compact=True` 的序列化会丢掉 `genome`（前沿里上千个点不该各带一份
        完整基因组）。**这里刻意不兜底造一个空基因组** ——
        「静默造一个占位对象」比报错危险得多：它会让下游拿着一个基因全为 0 的
        假 kernel 继续算，而那个结果看起来是个合法的数。
        需要完整基因组时，请从 `space.json` / `baseline.json` 取，
        或直接用评估器里的 `Trial` 对象（见 `pipeline.build` 的做法）。
        """
        if "genome" not in d:
            raise ValueError(
                "这条记录是 `compact=True` 序列化的（没有 `genome` 字段），"
                "不能还原成完整 Trial —— 请改从 space.json 取基因组，"
                "或直接用评估器里的对象。")
        return cls(
            genome=KernelGenome.from_dict(d["genome"]),
            op=d["op"], searcher=d.get("searcher", "oracle"),
            index=d.get("index", -1),
            drift=d["drift"], error_vs_reference=d["error_vs_reference"],
            effective_bits=d["effective_bits"], declared_bits=d["declared_bits"],
            bits_bias=d["bits_bias"], bits_headroom=d["bits_headroom"],
            bits_ok=d["bits_ok"], feasible=d["feasible"],
            deliverable=d["deliverable"],
            objectives=tuple(d["objectives"]), score=d["score"],
            latency_us=d["latency_us"], speedup=d["speedup"],
            throughput_tflops=d["throughput_tflops"],
            reject_reason=d.get("reject_reason", ""), rank=d.get("rank", -1),
            on_frontier=d.get("on_frontier", False),
            dominated_by=d.get("dominated_by", 0),
        )


# --------------------------------------------------------------------------- #
# 归约深度 → √n 偏置
# --------------------------------------------------------------------------- #

def accum_steps(cfg: dict[str, Any], op: str, gen: KernelGenome) -> int:
    """这个 kernel 在 K（或行式算子的 N）方向上的**舍入步数**。

    两遍归约：块间顺序累加 `⌈dim/tile⌉` 步 + 块内规约树 `⌈log2 tile⌉` 层
    + 跨 warp 规约 `⌈log2 num_warps⌉` 层。
    在线 / Welford 单遍：**逐元素**递推 ⇒ 步数就是元素个数。
    """
    row = bool(cfg["ops"][op]["row_wise"])
    m, n, k = op_probe_shape(cfg, op)
    dim = n if row else k
    tile = gen.tile_n if row else gen.tile_k
    tile = max(1, tile)
    if mode_of(op, gen.scheduling_strategy) in ("online", "welford"):
        return dim
    chunks = -(-dim // tile)
    tree = int(math.ceil(math.log2(max(1, tile)))) + \
        int(math.ceil(math.log2(max(1, gen.num_warps))))
    return chunks + tree


# --------------------------------------------------------------------------- #
# 评估器
# --------------------------------------------------------------------------- #

class KernelEvaluator:
    """一个算子的评估器。**缓存挂在实例上**，所以「同一基因组只算一次」是真的。"""

    def __init__(self, cfg: dict[str, Any], op: str) -> None:
        self.cfg = cfg
        self.op = op
        self.num = NumericsCache(cfg, op)
        self.baseline = baseline_genome(cfg, op)
        self.baseline_out = self.num.get(self.baseline)
        self.base_cost = cost_metrics(cfg, op, self.baseline)
        self.floor = float(cfg["numerics"]["error_floor"])
        self.slack = float(cfg["numerics"]["precision_slack_bits"])
        self._store: dict[str, Trial] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------ 内部
    def _bits_gate(self, gen: KernelGenome, eff: float, bias: float
                   ) -> tuple[bool, float]:
        declared = SIGNIFICANT_BITS[gen.accum_dtype]
        headroom = eff - (declared - bias)
        return headroom >= -self.slack, headroom

    # ------------------------------------------------------------------ 公开
    def evaluate(self, gen: KernelGenome, *, searcher: str = "oracle",
                 index: int = -1) -> Trial:
        key = gen.digest()
        got = self._store.get(key)
        if got is not None:
            self.cache_hits += 1
            return got
        self.cache_misses += 1

        out = self.num.get(gen)
        exact = self.num.exact(gen)
        drift = drift_of(out, self.baseline_out, self.floor)
        err_ref = drift_of(out, self.num.ref, self.floor)
        err_exact = drift_of(out, exact, self.floor)
        eff = effective_bits(err_exact)
        bias = expected_bits_bias(accum_steps(self.cfg, self.op, gen))
        bits_ok, headroom = self._bits_gate(gen, eff, bias)

        c = cost_metrics(self.cfg, self.op, gen)
        speedup = (self.base_cost["latency_us"] / c["latency_us"]
                   if c["latency_us"] > 0 else float("inf"))
        obj = vector(speedup, c["throughput_tflops"], eff)

        t = Trial(
            genome=gen, op=self.op, searcher=searcher, index=index,
            drift=drift, error_vs_reference=err_ref, effective_bits=eff,
            declared_bits=SIGNIFICANT_BITS[gen.accum_dtype], bits_bias=bias,
            bits_headroom=headroom, bits_ok=bits_ok,
            feasible=False, deliverable=False, objectives=obj, score=0.0,
            latency_us=c["latency_us"], speedup=speedup,
            throughput_tflops=c["throughput_tflops"],
        )
        t.cost = c
        self._store[key] = t
        return t

    def space(self, genomes: Sequence[KernelGenome]) -> list[Trial]:
        self.space_all = [self.evaluate(g, index=i) for i, g in enumerate(genomes)]
        return self.space_all

    def diagnostics(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "baseline_genome": self.baseline.to_dict(),
            "baseline_latency_us": round(self.base_cost["latency_us"], 9),
            "baseline_bottleneck": self.base_cost["bottleneck"],
            "probe_shape": list(op_probe_shape(self.cfg, self.op)),
            "production_shape": list(op_production_shape(self.cfg, self.op)),
            "n_unique_trials": len(self._store),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "numerics": self.num.diagnostics(),
        }


def apply_gate(trials: Sequence[Trial], tau: float) -> None:
    """把门 1 / 门 2 的结论写回 Trial（`feasible` / `deliverable`）。

    门 2 **与 τ 无关** —— 它不看误差，只看声明与实现是否一致。
    这正是它存在的理由：**门 1 的宽松度不该影响「诚实性」的判定**。
    """
    for t in trials:
        t.feasible = t.drift <= tau
        t.deliverable = t.feasible and t.bits_ok
        t.reject_reason = "" if t.feasible else f"drift {t.drift:.3e} > τ {tau:g}"


def assign_scores(trials: Sequence[Trial], scale: dict[str, float],
                  strategy: str, tau: float, penalty_weight: float,
                  drift_scale: float) -> None:
    """按约束处理策略打分。

    - `reject`：不可行者 `score = -1.0`（**不是 `-inf`** —— `-inf` 会污染
      均值 / 标准差 / 归一化，`L6-graphopt` 的坑 257）；
    - `penalty`：`score = tchebycheff − w·max(0, drift−τ)/drift_scale`。
      ⚠️ 分母是**固定的 `drift_scale`**，不是 τ —— 除以 τ 会让惩罚在 τ→0 时
      退化成硬约束，于是「软约束不保证可行」这条结论永远暴露不出来。

    ⚠️ 门 2 **不参与打分**：它是诚实性检查，不是优化目标。
    把它塞进 score 会让「搜索器学会了绕开门 2」变成一个可优化的问题。
    """
    for t in trials:
        base = tchebycheff(t.objectives, scale)
        if strategy == "reject":
            t.score = base if t.feasible else -1.0
        else:
            viol = max(0.0, t.drift - tau) / drift_scale
            t.score = base - penalty_weight * viol


def space_scale(trials: Sequence[Trial]) -> dict[str, float]:
    """全空间唯一的归一化标尺（与 τ / 策略 / 搜索器无关）。

    ⚠️ 键名**必须就是目标名**。第一版返回 `max_speedup` 这种带前缀的键，
    而 `objective.normalized` 按目标名查表 —— 于是它在**第一次打分**时
    就 `KeyError`。这次崩得很响，但同样的错位如果只影响其中一个目标，
    它会**静默地把那一维归一化成 0**，而 0 看起来像一个合法的归一化值。
    """
    return {
        "speedup": max((t.speedup for t in trials), default=1.0),
        "throughput": max((t.throughput_tflops for t in trials), default=1.0),
        "effective_bits": max((t.effective_bits for t in trials), default=1.0),
    }


def op_digest(cfg: dict[str, Any], op: str) -> str:
    return digest_of("op", op, jsonish(cfg["ops"][op]))


def jsonish(d: Any) -> str:
    import json
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


__all__ = ["Trial", "KernelEvaluator", "baseline_genome", "accum_steps",
           "apply_gate", "assign_scores", "space_scale", "op_digest"]
