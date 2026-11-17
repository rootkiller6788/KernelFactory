"""三个算子 + 它们的**数值孪生**。

## 孪生到底在孪生什么

真实 kernel 的精度损失来自两处，本模块两处都真的模拟：

1. **输入量化**：`tf32` 把输入截到 11 位、`bf16` 截到 8 位才送进乘法器。
   这是误差的**主项**（相对误差 ≈ 2^−bits），与分块无关。
2. **累加舍入**：每一次加法都要落回 `accum_dtype`。这是**次项**，
   但它**与分块方式有关** —— `tile_k` 越大、`num_warps` 越多，
   规约树越深/越宽，舍入路径就不同。

⇒ 于是 `tile_k` / `num_warps` 是**精度基因**（次项），
`accum_dtype` 是**精度主基因**，而 `tile_m` / `tile_n` / `vector_width` /
`memory_layout` **对误差的影响恰好为零** —— 这不是偷懒，是真的：
它们改变的是访存与并行切分，不改变 K 方向上的加法树。

**「一个对误差没有影响的基因」必须被显式测出来并报出**（`genes.json`），
否则读者会以为 9 个基因都在参与那条硬约束。

## 两种形状，两个用途

- `probe`：数值孪生用的小形状（要够深才看得见累加误差）；
- `production`：成本模型用的产出形状（量级要真实）。

⚠️ **两个尺度不同是刻意的，也是必须写明的**：本机没有 GPU，
在 512³ 上真的跑一遍 Python 浮点要几分钟。所以「延迟」是**解析模型**在产出形状上算的，
「误差」是**真跑**在探测形状上量的。**把假设标成测量是这类报告最危险的失真。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .numerics import (
    FULL_BITS,
    SIGNIFICANT_BITS,
    pair_sum,
    round_to_bits,
    scaled_max_error,
    seq_sum,
)
from .rng import Rng, derive_seed

# --------------------------------------------------------------------------- #
# 算子规格
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OpSpec:
    name: str
    row_wise: bool
    probe: tuple[int, int, int]
    production: tuple[int, int, int]


def op_specs(cfg: dict[str, Any]) -> dict[str, OpSpec]:
    return {
        name: OpSpec(name, bool(d["row_wise"]),
                     tuple(d["probe"]), tuple(d["production"]))  # type: ignore[arg-type]
        for name, d in cfg["ops"].items()
    }


def op_names(cfg: dict[str, Any]) -> list[str]:
    return sorted(cfg["ops"])


# --------------------------------------------------------------------------- #
# 确定性输入
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Inputs:
    """算子的输入。`x` 对 matmul 是 A（M×K），对行式算子是 x（M×N）。"""

    m: int
    n: int
    k: int
    x: tuple[float, ...]
    b: tuple[float, ...] = ()          # matmul 的右矩阵
    gamma: tuple[float, ...] = ()      # layernorm 的缩放
    beta: tuple[float, ...] = ()       # layernorm 的平移


def build_inputs(cfg: dict[str, Any], op: str) -> Inputs:
    """**根种子只由 `cfg["seed"]` 与算子名派生**（附录 C 239：多放一个成分
    就多一条看不见的耦合）。"""
    spec = op_specs(cfg)[op]
    m, n, k = spec.probe
    rng = Rng(derive_seed(cfg["seed"], "inputs", op))
    scale = float(cfg["ops"][op]["input_scale"])
    x = tuple(rng.values(m * k if not spec.row_wise else m * n, scale))
    if spec.row_wise:
        g = tuple(rng.values(n, 1.0)) if op == "layernorm" else ()
        be = tuple(rng.values(n, 0.5)) if op == "layernorm" else ()
        return Inputs(m=m, n=n, k=k, x=x, gamma=g, beta=be)
    return Inputs(m=m, n=n, k=k, x=x, b=tuple(rng.values(k * n, scale)))


# --------------------------------------------------------------------------- #
# 规约：块内按 warp 切分 → 树形规约 → 跨 warp 树形规约 → 块间顺序累加
# --------------------------------------------------------------------------- #

def warp_chunked_sum(values: Sequence[float], tile: int, bits: int,
                     num_warps: int) -> float:
    """**分块 + 块内并行规约**。这是本孪生里唯一的求和结构，三个算子共用。

    - 块内：按 `num_warps` 切成若干份（每份一个 warp 的私有部分和），
      每份内部**树形规约**，跨 warp 再合并；
    - 块间：**顺序累加**（跨块没有并行度，只能串起来）。

    ## ⚠️ 跨 warp 的合并**必须是顺序的**（第一版写成树形，于是这个基因死了）

    `pair_sum` 是一棵**平衡二叉树**。把 n 个元素切成两半、各自建树、再把两棵子树
    用一次加法合起来 —— 得到的树与「对 n 个元素直接建树」**逐节点相同**。
    同理，切成 2^k 等份再合并，树形状一模一样。

    ⇒ 后果：用 1 个 warp 与用 2 个 warp 的 kernel **逐位相同**，
    实测 `num_warps` 对误差的效应**恰好为 0**（在全部 600 个反事实配对上）。
    那看起来像一条结论（「num_warps 是纯速度基因」），其实是一个**建模错误**：
    真实的跨 warp 规约走共享内存，warp 之间是**串起来**的，不是再建一棵树。

    改成顺序合并之后，4 个 warp 与 8 个 warp 给出不同的舍入路径，
    而 1 个与 2 个仍然相同（两棵子树顺序合并，第一层仍是 `r(0+x)=x` 与一次加法）
    —— **这个「≥4 才生效」的门槛也是被量出来的，不是被假设的。**
    """
    n = len(values)
    if n == 0:
        return 0.0
    tile = max(1, min(tile, n))
    parts: list[float] = []
    for c in range(0, n, tile):
        chunk = list(values[c:c + tile])
        w = max(1, min(num_warps, len(chunk)))
        size = -(-len(chunk) // w)                    # ceil
        partials = [pair_sum(chunk[i:i + size], bits)
                    for i in range(0, len(chunk), size)]
        parts.append(seq_sum(partials, bits))
    return seq_sum(parts, bits)


# --------------------------------------------------------------------------- #
# 单算子：参考实现 / 算法精确值 / 内核模拟
# --------------------------------------------------------------------------- #

def reference(cfg: dict[str, Any], op: str, ins: Inputs) -> list[float]:
    """**fp64 参考实现**：教科书式、不模拟任何硬件细节。

    它是「这个算子在数学上应该等于什么」。内核与它的偏差是**绝对精度**；
    内核与**基线**的偏差是**一致性**（那条硬约束用的就是这个）。
    两个读数都要有 —— 只看一个会把「跟基线一起错」读成「对」。
    """
    if op == "matmul":
        out: list[float] = []
        for i in range(ins.m):
            for j in range(ins.n):
                acc = 0.0
                for kk in range(ins.k):
                    acc += ins.x[i * ins.k + kk] * ins.b[kk * ins.n + j]
                out.append(acc)
        return out
    if op == "softmax":
        out = []
        for i in range(ins.m):
            row = list(ins.x[i * ins.n:(i + 1) * ins.n])
            mx = max(row)
            ex = [math.exp(v - mx) for v in row]
            s = math.fsum(ex)
            out.extend(v / s for v in ex)
        return out
    if op == "layernorm":
        out = []
        for i in range(ins.m):
            row = list(ins.x[i * ins.n:(i + 1) * ins.n])
            mean = math.fsum(row) / ins.n
            var = math.fsum((v - mean) ** 2 for v in row) / ins.n
            inv = 1.0 / math.sqrt(var + float(cfg["ops"]["layernorm"]["eps"]))
            for j in range(ins.n):
                out.append((row[j] - mean) * inv * ins.gamma[j] + ins.beta[j])
        return out
    raise ValueError(f"未知算子：{op}")


def _matmul_kernel(ins: Inputs, tile_m: int, tile_n: int, tile_k: int,
                   num_warps: int, bits: int, mode: str, *, quantize: bool,
                   drop_tail: bool = False) -> list[float]:
    """matmul 的分块内核。

    ⚠️ **`mode` 也必须在这里生效。** 第一版 matmul 完全忽略 `mode`，
    而 `accum_steps` 却按 `persistent → online` 给了它一个不同的 √n 偏置 ——
    又是一处「声明与实现不符」。现在 `persistent`（在线）走**顺序累加**：
    persistent kernel 把每个输出元素的累加器常驻寄存器，沿 K 维一路加下去，
    没有块内并行规约。这是它与两遍分块归约的**真实**结构差别。
    """
    if quantize:
        a = [round_to_bits(v, bits) for v in ins.x]
        b = [round_to_bits(v, bits) for v in ins.b]
    else:
        a, b = list(ins.x), list(ins.b)
    out = [0.0] * (ins.m * ins.n)
    for i in range(ins.m):
        if drop_tail and i >= (ins.m // tile_m) * tile_m:
            break                       # 尾巴整块丢掉（非法变体）
        for j in range(ins.n):
            prods = [a[i * ins.k + kk] * b[kk * ins.n + j] for kk in range(ins.k)]
            if mode == "online":
                out[i * ins.n + j] = seq_sum(prods, bits)
            else:
                out[i * ins.n + j] = warp_chunked_sum(prods, tile_k, bits,
                                                      num_warps)
    return out


def _softmax_kernel(cfg: dict[str, Any], ins: Inputs, tile_n: int,
                    num_warps: int, bits: int, mode: str, *,
                    quantize: bool, drop_tail: bool = False,
                    skip_norm: bool = False) -> list[float]:
    out = [0.0] * (ins.m * ins.n)
    for i in range(ins.m):
        row = list(ins.x[i * ins.n:(i + 1) * ins.n])
        if quantize:
            row = [round_to_bits(v, bits) for v in row]
        if drop_tail:
            keep = (len(row) // tile_n) * tile_n
            row = row[:keep]
        if mode == "online":
            # 在线（flash 风格）：单遍维护 running max/sum，遇到更大的 max 就整体重标定。
            mx, s = -math.inf, 0.0
            for v in row:
                nm = max(mx, v)
                s = round_to_bits(s * math.exp(mx - nm) + math.exp(v - nm), bits) \
                    if mx > -math.inf else round_to_bits(math.exp(v - nm), bits)
                mx = nm
            mx_r = round_to_bits(mx, bits)
            denom = s
        else:
            # 两遍：先 max（按 tile_n 分块、块内树形规约），再 exp-sum。
            parts = []
            for c in range(0, len(row), tile_n):
                chunk = row[c:c + tile_n]
                w = max(1, min(num_warps, len(chunk)))
                size = -(-len(chunk) // w)
                partials = [max(chunk[t:t + size]) for t in range(0, len(chunk), size)]
                parts.append(max(partials))
            mx_r = max(parts) if parts else 0.0
            ex = [round_to_bits(math.exp(v - mx_r), bits) for v in row]
            denom = warp_chunked_sum(ex, tile_n, bits, num_warps)
        ex2 = [round_to_bits(math.exp(v - mx_r), bits) for v in row]
        if skip_norm:
            out[i * ins.n:i * ins.n + len(ex2)] = ex2     # 不除以分母（非法变体）
        else:
            out[i * ins.n:i * ins.n + len(ex2)] = [
                round_to_bits(v / denom, bits) for v in ex2]
    return out


def _layernorm_kernel(cfg: dict[str, Any], ins: Inputs, tile_n: int,
                      num_warps: int, bits: int, mode: str, *,
                      quantize: bool, drop_tail: bool = False,
                      no_center: bool = False, eps_zero: bool = False
                      ) -> list[float]:
    eps = 0.0 if eps_zero else float(cfg["ops"]["layernorm"]["eps"])
    out = [0.0] * (ins.m * ins.n)
    for i in range(ins.m):
        row = list(ins.x[i * ins.n:(i + 1) * ins.n])
        if quantize:
            row = [round_to_bits(v, bits) for v in row]
        if drop_tail:
            keep = (len(row) // tile_n) * tile_n
            row = row[:keep]
        if mode == "welford":
            # 单遍 Welford：均值与二阶矩同时递推（数值上与两遍不同）。
            cnt, mean, m2 = 0, 0.0, 0.0
            for v in row:
                cnt += 1
                d = v - mean
                mean = round_to_bits(mean + d / cnt, bits)
                m2 = round_to_bits(m2 + d * (v - mean), bits)
            var = round_to_bits(m2 / cnt, bits) if cnt else 0.0
        else:
            parts = []
            for c in range(0, len(row), tile_n):
                chunk = row[c:c + tile_n]
                w = max(1, min(num_warps, len(chunk)))
                size = -(-len(chunk) // w)
                partials = [math.fsum(chunk[t:t + size])
                            for t in range(0, len(chunk), size)]
                parts.append(sum(partials))
            mean = round_to_bits(sum(parts) / len(row), bits) if row else 0.0
            sq = [(v - mean) ** 2 for v in row]
            var = round_to_bits(
                warp_chunked_sum(sq, tile_n, bits, num_warps) / len(row), bits)
        inv = 1.0 / math.sqrt(max(var, 0.0) + eps)
        for j in range(len(row)):
            centered = row[j] if no_center else row[j] - mean
            out[i * ins.n + j] = round_to_bits(
                round_to_bits(centered * inv, bits) * ins.gamma[j] + ins.beta[j],
                bits)
    return out


# --------------------------------------------------------------------------- #
# 统一入口 + 缓存
# --------------------------------------------------------------------------- #

#: `scheduling_strategy` → 归约模式。**这是一条建模选择，不是测量**：
#: persistent kernel 把行状态常驻寄存器 ⇒ 天然对应在线/单遍归约；
#: 其余调度都是两遍。写在代码里而不是配置里，因为它不是可调阈值，
#: 而是「这三种调度分别长什么样」的定义。
MODE_BY_SCHEDULING: dict[str, str] = {
    "default": "two_pass",
    "pipelined": "two_pass",
    "persistent": "online",
}

#: 在线归约在**具体算子上**的名字。
#:
#: ⚠️ 第一版只有 `MODE_BY_SCHEDULING`，而 `_layernorm_kernel` 判的是
#: `mode == "welford"` —— 于是 **layernorm 的 `persistent` 调度跑的是两遍归约**，
#: 而 `accum_steps` 按在线归约（`dim` 步）算出了 √n 偏置。
#: 后果：`effective_bits` 的解析预测与实测差 **2.69 位**，
#: 而那条期望（「√n 偏置在 1 位以内」）被记成「被推翻」——
#: **它看起来像一条模型精度不足的结论，其实是一个「声明与实现不符」的 bug。**
#: 这恰好是本项目要抓的那一类错误，只不过这次抓到的是自己。
ONLINE_MODE_BY_OP: dict[str, str] = {
    "matmul": "online",
    "softmax": "online",
    "layernorm": "welford",
}


def mode_of(op: str, scheduling_strategy: str) -> str:
    """`(算子, 调度)` → 归约模式。**缓存键与数值函数必须走同一个入口。**"""
    m = MODE_BY_SCHEDULING[scheduling_strategy]
    return ONLINE_MODE_BY_OP[op] if m == "online" else m


#: 每个算子支持的非法变体（名字 → 说明）。**这是被测量的实验对象，不是对抗用例。**
#: ⚠️ `no_quantize_declared_fp32` **必须三个算子都有**：
#: 它是本项目的**头条对照组**（「声称 fp32、实际 bf16」vs「诚实的 bf16」），
#: 而它第一版只挂在 matmul 上 —— 于是另外两个算子的 `cross_op_headline`
#: 静默返回 `available=False`，报告里那一节直接消失，**不报错**。
ILLEGAL_VARIANTS: dict[str, tuple[str, ...]] = {
    "matmul": ("drop_tail_rows", "no_quantize_declared_fp32"),
    "softmax": ("skip_normalization", "drop_tail", "no_quantize_declared_fp32"),
    "layernorm": ("no_centering", "eps_zero", "drop_tail",
                  "no_quantize_declared_fp32"),
}


def kernel_output(cfg: dict[str, Any], op: str, gen: Any, ins: Inputs,
                  *, bits_override: int | None = None,
                  variant: str | None = None) -> list[float]:
    """按基因真的跑一遍内核。`variant` 非 None 时走**非法实现**。"""
    bits = (SIGNIFICANT_BITS[gen.accum_dtype] if bits_override is None
            else int(bits_override))
    mode = mode_of(op, gen.scheduling_strategy)
    quantize = bits_override is None
    kw: dict[str, Any] = {"quantize": quantize}
    if variant == "drop_tail_rows" or variant == "drop_tail":
        kw["drop_tail"] = True
    if variant == "skip_normalization":
        kw["skip_norm"] = True
    if variant == "no_centering":
        kw["no_center"] = True
    if variant == "eps_zero":
        kw["eps_zero"] = True
    if variant == "no_quantize_declared_fp32":
        # 声称 fp32（24 位）实际按 bf16 的 8 位跑 —— **声明与实现不符**。
        kw["quantize"] = True
        bits = SIGNIFICANT_BITS["bf16"]
    if op == "matmul":
        return _matmul_kernel(ins, gen.tile_m, gen.tile_n, gen.tile_k,
                              gen.num_warps, bits, mode, **kw)
    if op == "softmax":
        return _softmax_kernel(cfg, ins, gen.tile_n, gen.num_warps, bits,
                               mode, **kw)
    if op == "layernorm":
        return _layernorm_kernel(cfg, ins, gen.tile_n, gen.num_warps, bits,
                                 mode, **kw)
    raise ValueError(f"未知算子：{op}")


def algorithm_exact(cfg: dict[str, Any], op: str, gen: Any,
                    ins: Inputs) -> list[float]:
    """**同一个算法**在 fp64 下重跑（不量化、不舍入）。

    它是「有效位宽」测量的分母：内核与它的偏差只可能来自**精度**，
    不可能来自算法差异 —— 因为分块、调度、归约树全都一样。
    """
    if op == "matmul":
        return _matmul_kernel(ins, gen.tile_m, gen.tile_n, gen.tile_k,
                              gen.num_warps, FULL_BITS,
                              mode_of(op, gen.scheduling_strategy),
                              quantize=False)
    if op == "softmax":
        return _softmax_kernel(cfg, ins, gen.tile_n, gen.num_warps, FULL_BITS,
                               mode_of(op, gen.scheduling_strategy),
                               quantize=False)
    if op == "layernorm":
        return _layernorm_kernel(cfg, ins, gen.tile_n, gen.num_warps, FULL_BITS,
                                 mode_of(op, gen.scheduling_strategy),
                                 quantize=False)
    raise ValueError(f"未知算子：{op}")


#: 缓存键必须包含的基因 —— **逐个算子显式声明**，不靠「记得加」。
#:
#: 为什么需要一个模块级常量而不是把它写在 `_key` 里：`invariants.py` 有一条
#: 对抗用例会 **AST 扫 `kernel_output` 读了哪些 `gen.<属性>`**，然后断言
#: 「读了的」⊆「键里的」。于是「往数值函数里加一个基因、忘了加进缓存键」
#: 会**立刻红**，而不是变成一个命中率很漂亮、结果却错了的缓存。
#:
#: ⚠️ 合法路径与变体路径的成分**不一样**：`tile_m` / `tile_n` 只被
#: `drop_tail*` 变体读，合法路径下它们对输出**没有任何影响**。
#: 把它们放进合法键会让键空间膨胀 10 倍，每次都是一次注定命中的未命中。
CACHE_KEY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "matmul": ("tile_k", "num_warps"),
    "softmax": ("tile_n", "num_warps"),
    "layernorm": ("tile_n", "num_warps"),
}

#: 变体路径额外需要的成分（`drop_tail*` 靠分块尺寸决定「尾巴从哪开始」）。
VARIANT_KEY_COMPONENTS: tuple[str, ...] = ("tile_m", "tile_n")


def cache_key(op: str, gen: Any, bits: int, variant: str | None
              ) -> tuple[Any, ...]:
    """缓存键 = **恰好那些真的影响输出的成分**。"""
    comps = list(CACHE_KEY_COMPONENTS[op])
    if variant:
        comps += [c for c in VARIANT_KEY_COMPONENTS if c not in comps]
    red = tuple(getattr(gen, c) for c in comps)
    return (op, red, mode_of(op, gen.scheduling_strategy), int(bits), variant or "")


class NumericsCache:
    """按 `(op, 归约参数, bits)` 缓存内核输出。

    ## 为什么可以缓存，以及缓存键该长什么样

    **内核输出只由「归约结构 + 精度」决定**，与 `memory_layout` /
    `vector_width` / `num_stages` 无关（它们不改变加法树）。
    于是 1000 个基因组只有几十个不同的输出 —— **缓存命中率本身就是一条读数**：
    命中率接近 1 说明**绝大部分基因对数值结果没有影响**。

    ⚠️ 缓存键必须包含**全部**影响输出的成分。第一版漏了 `mode`
    （`persistent` 与 `default` 的归约不同），于是 `persistent` 的 kernel
    拿到了 `default` 的输出 —— 缓存命中率很漂亮，结果是错的。
    **漏一个成分的缓存，与一个正确的缓存，在命中率上长得一模一样。**
    """

    def __init__(self, cfg: dict[str, Any], op: str) -> None:
        self.cfg = cfg
        self.op = op
        self.ins = build_inputs(cfg, op)
        self.ref = reference(cfg, op, self.ins)
        self._store: dict[tuple[Any, ...], list[float]] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, gen: Any, bits: int, variant: str | None) -> tuple[Any, ...]:
        return cache_key(self.op, gen, bits, variant)

    def get(self, gen: Any, *, bits_override: int | None = None,
            variant: str | None = None) -> list[float]:
        bits = (SIGNIFICANT_BITS[gen.accum_dtype] if bits_override is None
                else int(bits_override))
        key = self._key(gen, bits, variant)
        got = self._store.get(key)
        if got is None:
            self.misses += 1
            got = kernel_output(self.cfg, self.op, gen, self.ins,
                                bits_override=bits_override, variant=variant)
            self._store[key] = got
        else:
            self.hits += 1
        return got

    def exact(self, gen: Any) -> list[float]:
        key = ("exact",) + self._key(gen, FULL_BITS, None)
        got = self._store.get(key)
        if got is None:
            self.misses += 1
            got = algorithm_exact(self.cfg, self.op, gen, self.ins)
            self._store[key] = got
        else:
            self.hits += 1
        return got

    def diagnostics(self) -> dict[str, Any]:
        n = self.hits + self.misses
        return {
            "op": self.op,
            "unique_outputs": self.misses,
            "cache_hits": self.hits,
            "hit_rate": (self.hits / n) if n else 0.0,
            "note": "内核输出只由「归约结构 + 精度」决定；命中率高 = "
                    "多数基因对数值结果没有影响。",
        }


def drift_of(got: Sequence[float], want: Sequence[float],
             floor: float) -> float:
    """**按问题量级归一化的最大绝对偏差**（见 `numerics.scaled_max_error`）。

    ⚠️ 项目里全部「误差」都走这一个入口 —— 三条读数
    （`drift` 对基线 / `error_vs_reference` 对 fp64 / `err_exact` 对同算法 fp64）
    用**同一个口径**，否则「门 1 的读数」与「有效位宽的读数」不可比，
    而它们会被并排印在同一张表里。
    """
    return scaled_max_error(list(got), list(want), floor)


__all__ = [
    "OpSpec", "Inputs", "op_specs", "op_names", "build_inputs", "reference",
    "kernel_output", "algorithm_exact", "NumericsCache", "drift_of",
    "warp_chunked_sum", "MODE_BY_SCHEDULING", "ONLINE_MODE_BY_OP",
    "mode_of", "ILLEGAL_VARIANTS",
]
