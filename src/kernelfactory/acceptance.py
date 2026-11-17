"""验收门槛：达标线 vs 实测。**每一条都必须能红。**

## 门槛 vs 对抗用例（`invariants.py`）—— 两层，不能混成一个词

- **门槛**（本文件）：**可调**的达标线，阈值写在 `config.acceptance` 里，
  可以被调松。红了意味着「这次没达标」。
- **对抗用例**（`invariants.py`）：**不可调**的不变量断言，配置里没有它的阈值。
  红了意味着**代码或产物自相矛盾**。

⚠️ 本项目还有一个更容易混的东西：`adversarial.json` 里的**非法变体**
（`drop_tail` / `skip_normalization` / `mantissa_truncate(b)` …）——
它们是**被测量的实验对象**（数据），既不是门槛也不是对抗用例。
**同一个词指了两个不同的东西，读者没法从报告内部发现**（附录 C 301）。

## 三条写门槛的纪律

1. **一条恒真的判据与没有判据，在输出里长得一模一样。** 写之前问：
   它在什么情况下会红？
2. **一条从不触发的保险丝与没有保险丝无法区分。** 所以保险丝自己也要有读数
   （`min_acceptance_checks` 断言验收表至少有这么长）。
3. **凡「有它在兜着」的量，都要算出来并断言。** `every_threshold_is_wired`
   用 AST 扫源码，确认配置里每个门槛键**真的被某段代码读过**。
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Callable, Sequence


_CHECKS: dict[str, Callable[[dict[str, Any], Any], tuple[Any, bool, str]]] = {}


def _check(name: str):
    def deco(fn):
        _CHECKS[name] = fn
        return fn
    return deco


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _ops(p: dict[str, Any]) -> list[str]:
    return sorted(p["config.json"]["resolved"]["ops"])


def _space(p: dict[str, Any], op: str) -> dict[str, Any]:
    return p["space.json"]["per_op"][op]


def _cells(p: dict[str, Any], op: str, strategy: str, tau: float
           ) -> dict[str, Any]:
    key = f"{strategy}@{tau:g}"
    return p["ladder.json"]["per_op"][op][key]["cells"]


# --------------------------------------------------------------------- 门槛
@_check("min_ops")
def _c_ops(p, thr):
    n = len(_ops(p))
    return n, n >= int(thr), f"{n} 个算子（要求 ≥ {thr}）"


@_check("min_space_per_op")
def _c_space(p, thr):
    bad = {op: _space(p, op)["n_genomes"] for op in _ops(p)
           if _space(p, op)["n_genomes"] < int(thr)}
    n = min(_space(p, op)["n_genomes"] for op in _ops(p)) if _ops(p) else 0
    return (n, not bad, f"最小的空间 {n} 个基因组（要求每个算子 ≥ {thr}）"
                        + (f"；不达标 {bad}" if bad else ""))


@_check("min_distinct_objective_vectors")
def _c_ov(p, thr):
    n = min(_space(p, op)["n_distinct_objective_vectors"] for op in _ops(p))
    return n, n >= int(thr), f"最小的「不同目标向量个数」{n}（要求 ≥ {thr}）"


@_check("max_structural_failures")
def _c_struct(p, thr):
    n = sum(_space(p, op)["n_structural_failures"] for op in _ops(p))
    return n, n <= int(thr), f"结构失败 {n} 次（要求 ≤ {thr}）"


@_check("min_distinct_accum_dtypes")
def _c_dtype(p, thr):
    n = min(len(_space(p, op)["accum_dtype_cardinality"]) for op in _ops(p))
    return (n, n >= int(thr),
            f"最小的精度档个数 {n}（要求 ≥ {thr}）—— "
            "**只有一档等于没有精度轴**，`Error(K) < ε` 会变成常数")


@_check("min_distinct_effective_bits")
def _c_ebits(p, thr):
    n = min(_space(p, op)["n_distinct_effective_bits"] for op in _ops(p))
    return (n, n >= int(thr),
            f"最小的「不同有效位宽个数」{n}（要求 ≥ {thr}）—— "
            "它是「有效位宽真的被测出来了」而不是「等于声明位宽」的读数")


@_check("min_ladder_speedup_gain")
def _c_gain(p, thr):
    """阶梯买到多少速度。**本项目与 L6-graphopt 的分水岭。**"""
    rows = []
    for op in _ops(p):
        for r in p["ladder.json"]["monotonicity"]:
            if r["op"] == op and r["strategy"] == "reject":
                sp = [x for x in r["best_speedup_oracle"] if x is not None]
                if len(sp) >= 2 and sp[0] > 0:
                    rows.append((op, sp[-1] / sp[0] - 1.0))
    if not rows:
        return None, False, "没有任何算子有可比的加速比序列"
    worst = min(g for _o, g in rows)
    return (worst, worst >= float(thr),
            f"最差算子上的相对增益 {worst:.4f}（要求 ≥ {thr}）；"
            f"{[(o, round(g, 4)) for o, g in rows]}")


@_check("min_precision_gate_rejections")
def _c_gate2(p, thr):
    n = sum(len(p["adversarial.json"]["per_op"][op]["gate1_only_anywhere"])
            for op in _ops(p))
    return (n, n >= int(thr),
            f"门 1 放行、门 2 拒掉的非法 kernel 共 {n} 个（要求 ≥ {thr}）—— "
            "0 个意味着门 2 是装饰")


@_check("min_frontier_points")
def _c_front(p, thr):
    """前沿的**有效大小**：不同的目标向量个数。

    按「基因组个数」量会被共线的第三轴灌水 —— `throughput` 与 `speedup`
    完全共线，同一个权衡点会被算两次。门槛落在**去重后的个数**上。
    """
    n = min(p["ladder.json"]["per_op"][op]["reject@1"]["cells"]["oracle"]
            ["n_frontier_distinct"] for op in _ops(p))
    raw = min(p["ladder.json"]["per_op"][op]["reject@1"]["cells"]["oracle"]
              ["n_frontier"] for op in _ops(p))
    return (n, n >= int(thr),
            f"最小前沿 {raw} 个点 / **{n} 个不同权衡点**（要求 ≥ {thr}）")


@_check("max_greedy_gap_vs_oracle")
def _c_greedy(p, thr):
    """**greedy 交出的答案**相对 oracle 的延迟缺口（在**默认容差**下）。

    ⚠️ 口径选的是「**答案的延迟**」，不是「超体积」。第一版量的是 τ=1.0 时的
    超体积缺口，实测 0.57~0.61 —— 那**不是 greedy 坏了**，而是它在做另一件事：
    greedy 沿**一个**标量化目标爬山，所以它只覆盖前沿上的**一个**点，
    而超体积量的是**整条前沿**的覆盖度。**用一个标量化搜索器去量前沿覆盖度，
    量到的是「它的设计目标不是这个」，不是「它弱」。**

    改量答案的延迟之后，缺口落在 0.02 附近 —— 那才是「搜索器有没有找到好答案」。
    前沿覆盖度仍然报在报告里（见 `searcher_front_coverage`），只是不当门槛。
    """
    cfg = p["config.json"]["resolved"]
    tol = float(cfg["semantics"]["tolerance"])
    worst = 0.0
    who = None
    rows = {}
    for op in _ops(p):
        c = _cells(p, op, "reject", tol)
        o = c["oracle"]["best"]
        g = c["greedy"]["best"]
        if o is None or g is None:
            return None, False, f"{op} 在默认容差下 oracle 或 greedy 没有答案"
        gap = g["latency_us"] / o["latency_us"] - 1.0
        rows[op] = {"oracle_latency_us": round(o["latency_us"], 9),
                    "greedy_latency_us": round(g["latency_us"], 9),
                    "gap": round(gap, 9)}
        if gap > worst:
            worst, who = gap, op
    return (worst, worst <= float(thr),
            f"默认容差 τ={tol:g} 下 greedy 相对 oracle 的最大延迟缺口 {worst:.6f}"
            f"（{who}，要求 ≤ {thr}）；{rows}")


@_check("oracle_is_optimal")
def _c_oracle(p, thr):
    worse = []
    for op in _ops(p):
        c = _cells(p, op, "reject", 1.0)
        o = c["oracle"]["hypervolume"]
        for name, cell in c.items():
            if name == "oracle":
                continue
            if cell["hypervolume"] > o + 1e-12:
                worse.append((op, name, cell["hypervolume"], o))
    return (bool(worse), (not worse) == bool(thr),
            f"超过 oracle 超体积的搜索器 {worse}")


@_check("max_answer_infeasible_rate")
def _c_answer(p, thr):
    bad = 0
    tot = 0
    for op in _ops(p):
        for key, rung in p["ladder.json"]["per_op"][op].items():
            for name, cell in rung["cells"].items():
                b = cell.get("best")
                tot += 1
                if b is not None and not b["feasible"]:
                    bad += 1
    rate = (bad / tot) if tot else 0.0
    return rate, rate <= float(thr), (f"交出去的答案不可行率 {rate:.6f}"
                                      f"（{bad}/{tot}，要求 ≤ {thr}）")


@_check("max_answer_dishonest_rate")
def _c_dishonest(p, thr):
    """**门 2 的交付判据**：交出去的答案必须诚实。

    与 `max_answer_infeasible_rate` 分开是刻意的：
    「可行」与「诚实」是两件事，合成一个数就再也分不开了。
    """
    bad = 0
    tot = 0
    for op in _ops(p):
        for _key, rung in p["ladder.json"]["per_op"][op].items():
            for _name, cell in rung["cells"].items():
                b = cell.get("best")
                tot += 1
                if b is not None and not b["bits_ok"]:
                    bad += 1
    rate = (bad / tot) if tot else 0.0
    return rate, rate <= float(thr), (f"交出去的答案不诚实率 {rate:.6f}"
                                      f"（{bad}/{tot}，要求 ≤ {thr}）")


@_check("min_monotone_strategies")
def _c_mono(p, thr):
    rows = p["ladder.json"]["monotonicity"]
    n = sum(1 for r in rows if r["monotone"])
    return n, n >= int(thr), f"{n}/{len(rows)} 行阶梯单调（要求 ≥ {thr}）"


@_check("min_adversarial_cases")
def _c_cases(p, thr):
    """**对抗用例条数的保险丝** —— 删用例会让它红。

    读的是 `invariants` 的**注册表大小**，不跑用例：
    于是它能在 `cases` 被填进 `acceptance.json` **之前**生效
    （`cases` 依赖 `summary.json`，而 `summary.json` 依赖验收结果 ⇒ 不能循环）。

    ⚠️ 口径：这里数的**不是** `adversarial.json` 里的非法变体（那些是数据）。
    """
    from .invariants import registered_cases
    n = len(registered_cases())
    return (n, n >= int(thr),
            f"注册了 {n} 条对抗用例（要求 ≥ {thr}）—— 它们是不可调的不变量断言，"
            f"与 `adversarial.json` 里那些**非法变体**不是一回事")


@_check("min_acceptance_checks")
def _c_self(p, thr):
    """**保险丝的自指读数**：验收表本身的长度。删门槛会让它红。

    ⚠️ 它**不能**去读 `acceptance.json`（那正是正在构造的产物，会读到半成品）。
    所以它数配置里的键，再加 2 —— 加的那 2 条是常开的结构性保险丝，
    它们不在配置里（因此不许被调松）。
    """
    n = len(p["config.json"]["resolved"]["acceptance"]) + 2
    return n, n >= int(thr), f"验收表共 {n} 条（含本条与两条常开保险丝，要求 ≥ {thr}）"


# --------------------------------------------------------------------- 接线
def string_literals(pkg_dir: Path, exclude: Sequence[str] = ()) -> set[str]:
    """包内全部 `.py` 的字符串字面量，**排除文档字符串与 `exclude`**。

    - 排除 `config.py`：`DEFAULTS` 里的键本身就是字符串字面量，
      把 `config.py` 算进来，这条检查会**恒真**。
    - 排除文档字符串：设计意图就写在文档字符串里，把「解释」当成「接线」是自欺。
      实现上用 `ast` **节点身份**排除（不是文本匹配）—— 同一个字符串既当
      文档字符串又当真字面量时，文本匹配会误判。
    """
    out: set[str] = set()
    excluded = set(exclude)
    for path in sorted(pkg_dir.rglob("*.py")):
        if path.name in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc_nodes: set[int] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                doc_nodes.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in doc_nodes):
                out.add(node.value)
    return out


def wiring_report(cfg: dict[str, Any], pkg_dir: Path) -> dict[str, Any]:
    """三类接线核对，每类都可能红。"""
    lits = string_literals(pkg_dir, exclude=("config.py",))
    acc_keys = sorted(cfg["acceptance"])
    unwired = [k for k in acc_keys if k not in lits]

    from .expectations import _EVALUATORS
    exp_declared = set(cfg["expectations"])
    exp_orphan = sorted(set(_EVALUATORS) - exp_declared)
    exp_missing = sorted(exp_declared - set(_EVALUATORS))

    from .search import declared_params
    declared = declared_params()
    search_keys = set(cfg["search"])
    bad_params = sorted({f"{s}.{k}" for s, ks in declared.items()
                         for k in ks if k not in search_keys})

    return {
        "n_acceptance_keys": len(acc_keys),
        "unwired_acceptance_keys": unwired,
        "orphan_expectation_evaluators": exp_orphan,
        "missing_expectation_evaluators": exp_missing,
        "searcher_params_not_in_config": bad_params,
        "all_wired": not unwired and not exp_orphan and not exp_missing
                     and not bad_params,
        "note": "扫的是除 config.py 之外的全部源码；文档字符串不计入接线。",
    }


# --------------------------------------------------------------------- 主入口
def run_acceptance(products: dict[str, Any],
                   pkg_dir: Path | None = None) -> dict[str, Any]:
    cfg = products["config.json"]["resolved"]
    declared = cfg["acceptance"]

    checks: list[dict[str, Any]] = []
    for name in sorted(declared):
        thr = declared[name]
        fn = _CHECKS.get(name)
        if fn is None:
            checks.append({
                "name": name, "threshold": thr, "observed": None, "passed": False,
                "detail": "**没有任何检查函数认领这条门槛** —— 它看起来像防线，"
                          "其实不是",
            })
            continue
        observed, passed, detail = fn(products, thr)
        checks.append({
            "name": name, "threshold": thr,
            "observed": (None if observed is None
                         else (round(float(observed), 12)
                               if isinstance(observed, (int, float))
                               and not isinstance(observed, bool) else observed)),
            "passed": bool(passed), "detail": detail,
        })

    orphan = sorted(set(_CHECKS) - set(declared))
    if pkg_dir is None:
        pkg_dir = Path(__file__).resolve().parent
    wiring = wiring_report(cfg, pkg_dir)
    checks.append({
        "name": "every_threshold_is_wired", "threshold": True,
        "observed": wiring["all_wired"], "passed": bool(wiring["all_wired"]),
        "detail": ("全部门槛键都被源码读过、全部期望都有求值函数、"
                   "搜索器声明的超参都在配置里"
                   if wiring["all_wired"] else
                   f"未接线：{wiring['unwired_acceptance_keys']}；"
                   f"孤儿期望：{wiring['orphan_expectation_evaluators']}；"
                   f"缺求值函数：{wiring['missing_expectation_evaluators']}；"
                   f"声明了但配置里没有的超参：{wiring['searcher_params_not_in_config']}"),
    })

    # 常开保险丝：参考点必须**逐算子**唯一。
    # 按级取参考点会让「放宽容差」同时移动参考点，超体积的变化分不清来源
    # （`L6-graphopt` 实测凭空涨 68%，全是假象，附录 C 283）。
    #
    # ⚠️ 口径：**逐算子**。三个算子的空间不同 ⇒ 它们的 `min(各维) − pad` 本来就
    #    不一样。把三个算子的参考点并成一个集合去数「有几个」，
    #    量到的是「算子之间差多少」，不是「同一算子内各格子的参考点是否唯一」。
    refs_per_op: dict[str, list[list[float]]] = {}
    for op in _ops(products):
        refs_per_op[op] = sorted({tuple(r["reference"])
                                  for r in products["ladder.json"]["per_op"][op].values()})
    ref_ok = all(len(v) == 1 for v in refs_per_op.values())
    checks.append({
        "name": "reference_is_shared_across_rungs", "threshold": 1,
        "observed": max((len(v) for v in refs_per_op.values()), default=0),
        "passed": ref_ok,
        "detail": (f"逐算子的参考点都唯一：{refs_per_op}" if ref_ok else
                   f"有算子出现了多个参考点："
                   f"{ {o: v for o, v in refs_per_op.items() if len(v) != 1} }"
                   " —— 各用各的参考点，超体积之间没法比"),
    })

    return {
        "checks": checks,
        "n": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "failed": [c["name"] for c in checks if not c["passed"]],
        "orphan_check_functions": orphan,
        "wiring": wiring,
    }


__all__ = ["run_acceptance", "wiring_report", "string_literals"]
