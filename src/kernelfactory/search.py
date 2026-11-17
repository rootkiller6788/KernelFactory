"""搜索器：在基因空间里找 `arg min Latency subject to 门 1 ∧ 门 2`。

## 四条纪律（`L5-modeloptimizationfactory` 与 `L6-graphopt` 换来的）

1. **「省算力」与「花预算」是两件事**（附录 C 281）。
   `Visits.used` 按「本搜索器**首次**提案」计；自己重复提同一个基因组、
   或别人算过的，都不该扣自己的预算。混在一起会出现「提案 400 次、预算用了 0」。
2. **去重集合必须按「你要去重的东西」建**（附录 C 284）。这里按 `genome.digest()`。
3. **不可行者给 `-1.0` 而不是 `-inf`**（附录 C 257）——
   `-inf` 会污染均值 / 标准差 / 归一化。
4. **oracle 的预算 = 空间大小，其余 = 同一个预算**，这个**不对称必须显式报出来** ——
   否则 `oracle_is_optimal` 会从数学必然退化成「可能」。

## 邻域的定义

基因空间是笛卡尔积 ⇒ 一个基因组的**邻居 = 改一个基因**（改成该基因的任一档）。
这不是审美：**如果邻域是「随机换一个基因组」，那 greedy 与 random 就没有区别了**，
而「greedy 是短视的」这条结论也就无从谈起。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .genome import GENE_ORDER, KernelGenome, gene_values
from .rng import Rng, derive_seed, digest_of

#: 各搜索器**声明消费**的超参（`acceptance.wiring_report` 会核对它们
#: 真的存在于 `config.search` —— 一个写在文档里、没人读的超参
#: 与一个不存在的超参在输出里长得一样）。
_DECLARED: dict[str, tuple[str, ...]] = {
    "random": ("budget_per_searcher",),
    "greedy": ("budget_per_searcher",),
    "beam": ("budget_per_searcher", "beam_width"),
    "evolution": ("budget_per_searcher", "population", "mutation_rate"),
    "oracle": (),
}


def declared_params() -> dict[str, tuple[str, ...]]:
    return dict(_DECLARED)


# --------------------------------------------------------------------------- #
# 预算
# --------------------------------------------------------------------------- #

@dataclass
class Visits:
    """预算与命中分开计。**`used` 只被「本搜索器首次提案」推进。**"""

    budget: int
    used: int = 0
    self_repeats: int = 0
    shared: int = 0
    seen: set[str] = field(default_factory=set)

    def exhausted(self) -> bool:
        return self.used >= self.budget

    def take(self, digest: str) -> bool:
        """返回 `True` 表示这次提案要花预算（首次）。"""
        if digest in self.seen:
            self.self_repeats += 1
            return False
        self.seen.add(digest)
        self.used += 1
        return True

    def note_shared(self) -> None:
        self.shared += 1


# --------------------------------------------------------------------------- #
# 上下文
# --------------------------------------------------------------------------- #

@dataclass
class SearchContext:
    """搜索器看得见的一切。**它们看不到基因组本身之外的任何东西** ——
    没有成本模型、没有真值、没有容差。搜索器只能问「这个基因组的得分是多少」。"""

    op: str
    genomes: Sequence[KernelGenome]
    score_of: Callable[[KernelGenome], float]
    budget: int
    rng: Rng
    neighbors_of: Callable[[KernelGenome], list[KernelGenome]]
    values: dict[str, tuple[Any, ...]]

    def evaluate(self, gen: KernelGenome, visits: Visits) -> float:
        visits.take(gen.digest())
        return self.score_of(gen)


def make_context(cfg: dict[str, Any], op: str,
                 genomes: Sequence[KernelGenome],
                 score_of: Callable[[KernelGenome], float]) -> SearchContext:
    """建上下文。邻域**预计算**（空间不大，但每次都重算是浪费）。"""
    by_digest = {g.digest(): g for g in genomes}
    vals = gene_values(cfg)
    nbr_cache: dict[str, list[KernelGenome]] = {}

    def neighbors(gen: KernelGenome) -> list[KernelGenome]:
        key = gen.digest()
        got = nbr_cache.get(key)
        if got is not None:
            return got
        out: list[KernelGenome] = []
        for name in GENE_ORDER:
            for v in vals[name]:
                if v == getattr(gen, name):
                    continue
                cand = _with(gen, name, v)
                if cand.digest() in by_digest:
                    out.append(cand)
        out.sort(key=lambda g: g.digest())
        nbr_cache[key] = out
        return out

    return SearchContext(
        op=op, genomes=list(genomes), score_of=score_of,
        budget=int(cfg["search"]["budget_per_searcher"]),
        rng=Rng(derive_seed(cfg["seed"], "search", op)),
        neighbors_of=neighbors, values=vals,
    )


def _with(gen: KernelGenome, name: str, value: Any) -> KernelGenome:
    d = gen.to_dict()
    d[name] = value
    return KernelGenome.from_dict(d)


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #

@dataclass
class SearchResult:
    searcher: str
    op: str
    best: KernelGenome | None
    best_score: float
    evaluated: list[str]
    visits: Visits
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "searcher": self.searcher,
            "op": self.op,
            "best_genome": self.best.to_dict() if self.best else None,
            "best_label": self.best.label() if self.best else None,
            "best_score": round(self.best_score, 12),
            "n_evaluated_unique": len(self.evaluated),
            "budget": self.visits.budget,
            "budget_used": self.visits.used,
            "self_repeats": self.visits.self_repeats,
            "trace": self.trace,
        }


# --------------------------------------------------------------------------- #
# 搜索器
# --------------------------------------------------------------------------- #

def _finish(name: str, ctx: SearchContext, visits: Visits,
            best: KernelGenome | None, best_score: float,
            evaluated: list[str], trace: list[dict[str, Any]]) -> SearchResult:
    return SearchResult(searcher=name, op=ctx.op, best=best,
                        best_score=best_score, evaluated=evaluated,
                        visits=visits, trace=trace)


def search_random(ctx: SearchContext) -> SearchResult:
    """随机采样（**无放回**：有放回会让「预算 120」实际只看了 80 个不同点）。"""
    v = Visits(ctx.budget)
    order = sorted(ctx.genomes, key=lambda g: g.digest())
    ctx.rng.shuffle(order)
    best, bs, ev = None, float("-inf"), []
    for g in order:
        if v.exhausted():
            break
        s = ctx.evaluate(g, v)
        ev.append(g.digest())
        if s > bs:
            best, bs = g, s
    return _finish("random", ctx, v, best, bs, ev, [])


def search_greedy(ctx: SearchContext) -> SearchResult:
    """单步最优的爬山。**短视**：它只看一步的收益，不看两步之后。"""
    v = Visits(ctx.budget)
    cur = _seed(ctx)
    cs = ctx.evaluate(cur, v)
    ev, trace = [cur.digest()], []
    while not v.exhausted():
        moved = False
        for n in ctx.neighbors_of(cur):
            if v.exhausted():
                break
            s = ctx.evaluate(n, v)
            ev.append(n.digest())
            if s > cs:
                cur, cs, moved = n, s, True
        trace.append({"step": len(trace), "label": cur.label(),
                      "score": round(cs, 12), "improved": moved})
        if not moved:
            break
    return _finish("greedy", ctx, v, cur, cs, ev, trace)


def search_beam(ctx: SearchContext, width: int | None = None) -> SearchResult:
    """束搜索：每轮把全部束的邻居合起来取前 `width` 个。"""
    w = int(width or ctx.values and 4)
    v = Visits(ctx.budget)
    cur = _seed(ctx)
    cs = ctx.evaluate(cur, v)
    ev, trace = [cur.digest()], []
    beam = [(cs, cur)]
    best, bs = cur, cs
    while not v.exhausted():
        cand: dict[str, tuple[float, KernelGenome]] = {}
        for _s, g in beam:
            for n in ctx.neighbors_of(g):
                if v.exhausted():
                    break
                d = n.digest()
                # ⚠️ 只收**未访问过**的（同 greedy 的死循环修法）
                if d in cand or d in v.seen:
                    continue
                sc = ctx.evaluate(n, v)
                ev.append(d)
                cand[d] = (sc, n)
        if not cand:
            break
        ranked = sorted(cand.values(), key=lambda t: (-t[0], t[1].digest()))[:w]
        beam = ranked
        if ranked and ranked[0][0] > bs:
            bs, best = ranked[0][0], ranked[0][1]
        trace.append({"step": len(trace), "width": len(beam),
                      "best_score": round(bs, 12)})
    return _finish("beam", ctx, v, best, bs, ev, trace)


def search_evolution(ctx: SearchContext, population: int | None = None,
                     mutation_rate: float | None = None) -> SearchResult:
    """进化：`population` 个个体，每轮按得分取前一半，变异 + 交叉。

    ⚠️ 交叉**必须按基因位**（逐基因从两个父本里取），不能「随机挑一个父本」——
    后者是复制，不是交叉，而「进化 vs 随机」的差别会因此消失。
    """
    pop_n = int(population or 12)
    rate = float(mutation_rate if mutation_rate is not None else 0.35)
    v = Visits(ctx.budget)
    order = sorted(ctx.genomes, key=lambda g: g.digest())
    ctx.rng.shuffle(order)
    pop: list[tuple[float, KernelGenome]] = []
    ev: list[str] = []
    for g in order[:pop_n]:
        if v.exhausted():
            break
        s = ctx.evaluate(g, v)
        ev.append(g.digest())
        pop.append((s, g))
    pop.sort(key=lambda t: (-t[0], t[1].digest()))
    trace = [{"gen": 0, "best_score": round(pop[0][0], 12) if pop else None}]
    gen = 0
    #: 保险丝：代数上限。**它是必需的，不是保守** —— 当变异反复产不出新个体时，
    #: 循环的每一步都不推进预算（见 greedy 的说明），而「没推进」是看不见的。
    max_gen = 4 * ctx.budget
    while not v.exhausted() and len(pop) >= 2 and gen < max_gen:
        gen += 1
        before = v.used
        parents = pop[:max(2, len(pop) // 2)]
        children: list[tuple[float, KernelGenome]] = []
        attempts = 0
        while len(children) < pop_n and not v.exhausted() and attempts < 4 * pop_n:
            attempts += 1
            a = ctx.rng.choice([g for _s, g in parents])
            b = ctx.rng.choice([g for _s, g in parents])
            child = _crossover(a, b, ctx)
            if ctx.rng.random() < rate:
                child = _mutate(child, ctx)
            d = child.digest()
            s = ctx.evaluate(child, v)
            ev.append(d)
            children.append((s, child))
        merged = {g.digest(): (s, g) for s, g in pop + children}
        pop = sorted(merged.values(), key=lambda t: (-t[0], t[1].digest()))[:pop_n]
        trace.append({"gen": gen, "best_score": round(pop[0][0], 12),
                      "budget_used": v.used})
        if v.used == before:
            # 一整代没推进预算 ⇒ 再转下去也不会（变异空间已耗尽）
            break
    best, bs = (pop[0][1], pop[0][0]) if pop else (None, float("-inf"))
    return _finish("evolution", ctx, v, best, bs, ev, trace)


def search_oracle(ctx: SearchContext) -> SearchResult:
    """穷举。**预算 = 空间大小**（这不是特权，是定义）。"""
    v = Visits(len(ctx.genomes))
    best, bs, ev = None, float("-inf"), []
    for g in sorted(ctx.genomes, key=lambda x: x.digest()):
        s = ctx.evaluate(g, v)
        ev.append(g.digest())
        if s > bs:
            best, bs = g, s
    return _finish("oracle", ctx, v, best, bs, ev, [])


SEARCHERS: dict[str, Any] = {
    "random": search_random,
    "greedy": search_greedy,
    "beam": search_beam,
    "evolution": search_evolution,
    "oracle": search_oracle,
}


def run_searchers(cfg: dict[str, Any], ctx: SearchContext,
                  names: Sequence[str]) -> dict[str, SearchResult]:
    out: dict[str, SearchResult] = {}
    for n in names:
        fn = SEARCHERS[n]
        if n == "beam":
            out[n] = fn(ctx, width=int(cfg["search"]["beam_width"]))
        elif n == "evolution":
            out[n] = fn(ctx, population=int(cfg["search"]["population"]),
                        mutation_rate=float(cfg["search"]["mutation_rate"]))
        else:
            out[n] = fn(ctx)
    return out


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _seed(ctx: SearchContext) -> KernelGenome:
    """确定性的起点：按 digest 排序后的第一个。

    ⚠️ **不能用 `ctx.rng`** —— 那会让 greedy 的起点随调用顺序变，
    而 greedy 的所有结论（「短视」）都要在**同一起点**上才可比。
    """
    return min(ctx.genomes, key=lambda g: g.digest())


def _crossover(a: KernelGenome, b: KernelGenome, ctx: SearchContext
               ) -> KernelGenome:
    """逐基因位取父本之一。结果不保证在合法空间里 ⇒ 用邻居表兜底。"""
    d: dict[str, Any] = {}
    for name in GENE_ORDER:
        d[name] = getattr(a, name) if ctx.rng.random() < 0.5 else getattr(b, name)
    cand = KernelGenome.from_dict(d)
    if any(cand.digest() == g.digest() for g in ctx.genomes):
        return cand
    return a


def _mutate(gen: KernelGenome, ctx: SearchContext) -> KernelGenome:
    """改一个基因。改出来的值**必须在该基因的取值表里** —— 否则会造出
    一个合法空间之外的基因组，而它照样能被评估（成本模型不看合法性）。"""
    nbrs = ctx.neighbors_of(gen)
    if not nbrs:
        return gen
    return ctx.rng.choice(nbrs)


def permutation_counts(values: dict[str, tuple[Any, ...]]) -> int:
    """笛卡尔积大小（供对抗用例核对空间大小）。"""
    n = 1
    for v in values.values():
        n *= len(v)
    return n


__all__ = ["Visits", "SearchContext", "SearchResult", "make_context",
           "SEARCHERS", "run_searchers", "declared_params",
           "permutation_counts", "search_random", "search_greedy",
           "search_beam", "search_evolution", "search_oracle"]
