"""标定：**先量误差，再定阶梯**。

跑一遍全部算子 × 全部合法基因，打印：
1. 空间大小与被过滤原因；
2. 每个精度档的 drift 分布（min / p50 / max）；
3. 每个精度档的**有效位宽**与 `headroom = eff − (declared − √n偏置)`；
4. 建议的 `precision_slack_bits`（= 全部合法 kernel 的 headroom 最小值 − 余量）；
5. 每个容差档下的可行域大小（阶梯的落点）；
6. 耗时。

不落盘、不改配置 —— 它只**报数**，人看着数去改配置。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelfactory.config import DEFAULTS, dump_config          # noqa: E402
from kernelfactory.evaluate import KernelEvaluator, accum_steps, space_scale  # noqa: E402
from kernelfactory.genome import enumerate_space                # noqa: E402
from kernelfactory.numerics import SIGNIFICANT_BITS, effective_bits, expected_bits_bias  # noqa: E402
from kernelfactory.ops import drift_of                          # noqa: E402


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def main() -> int:
    cfg = json.loads(json.dumps(DEFAULTS))
    t0 = time.perf_counter()
    all_rows: list[dict] = []
    for op in sorted(cfg["ops"]):
        genomes, diag = enumerate_space(cfg, op)
        ev = KernelEvaluator(cfg, op)
        t = time.perf_counter()
        trials = ev.space(genomes)
        dt = time.perf_counter() - t
        print(f"\n===== {op} =====")
        print(f"空间 {diag['n_raw']} 原始 → {diag['n_valid']} 合法 "
              f"→ {diag['n_distinct']} 去重（砍 {diag['n_pruned']}，"
              f"折叠 {diag['n_collapsed_duplicates']}）")
        print(f"过滤原因：{diag['prune_reasons']}")
        print(f"基因取值个数：{diag['gene_cardinality']}")
        print(f"数值+成本评估 {dt:.2f} s（唯一 trial {ev.cache_misses}）")

        by_dtype: dict[str, list] = {}
        for t_ in trials:
            by_dtype.setdefault(t_.genome.accum_dtype, []).append(t_)
        print(f"{'dtype':6s} {'n':>5s} {'drift p50':>12s} {'drift max':>12s} "
              f"{'effbits p50':>12s} {'bias p50':>9s} {'headroom min':>13s}")
        for dt_, ts in sorted(by_dtype.items()):
            dr = [x.drift for x in ts]
            ef = [x.effective_bits for x in ts]
            bi = [x.bits_bias for x in ts]
            hd = [x.bits_headroom for x in ts]
            print(f"{dt_:6s} {len(ts):5d} {pct(dr,0.5):12.3e} {max(dr):12.3e} "
                  f"{pct(ef,0.5):12.4f} {pct(bi,0.5):9.3f} {min(hd):13.4f}")
            all_rows.append({"op": op, "dtype": dt_, "n": len(ts),
                             "drift_p50": pct(dr, 0.5), "drift_max": max(dr),
                             "eff_p50": pct(ef, 0.5), "bias_p50": pct(bi, 0.5),
                             "headroom_min": min(hd), "headroom_max": max(hd),
                             "declared": SIGNIFICANT_BITS[dt_]})

        # 不同 drift 值的个数（精度轴有多"粗"）
        nd = len({round(x.drift, 15) for x in trials})
        ne = len({round(x.effective_bits, 9) for x in trials})
        print(f"去重后：drift {nd} 个不同值 / effbits {ne} 个不同值 "
              f"/ 空间 {len(trials)}")
        print(f"输出缓存：{ev.num.diagnostics()}")

        # 阶梯落点
        tol = [float(x) for x in cfg["ladder"]["tolerances"]]
        feas = [(f"{t:g}", sum(1 for x in trials if x.drift <= t)) for t in tol]
        print("每级可行域：" + "  ".join(f"{k}:{v}" for k, v in feas))
        ok = [(f"{t:g}", sum(1 for x in trials if x.drift <= t and x.bits_ok))
              for t in tol]
        print("每级可行∧诚实：" + "  ".join(f"{k}:{v}" for k, v in ok))

    # ---- 尺子：mantissa_truncate(b) 的漂移 ----
    print("\n===== 尺子：mantissa_truncate(b)（声称 fp32，实际 b 位）=====")
    from kernelfactory.ops import ILLEGAL_VARIANTS, kernel_output
    from kernelfactory.evaluate import baseline_genome
    for op in sorted(cfg["ops"]):
        ev = KernelEvaluator(cfg, op)
        base = ev.baseline
        for b in [int(x) for x in cfg["adversarial"]["mantissa_bits"]]:
            out = kernel_output(cfg, op, base, ev.num.ins, bits_override=b)
            d = drift_of(out, ev.baseline_out, ev.num.floor if hasattr(ev.num, "floor") else 1e-12)
            print(f"  {op:10s} b={b:2d}  drift={d:.4e}  ≈2^-b={2.0**-b:.4e}")
        for v in ILLEGAL_VARIANTS[op]:
            out = kernel_output(cfg, op, base, ev.num.ins, variant=v)
            d = drift_of(out, ev.baseline_out, 1e-12)
            print(f"  {op:10s} 非法变体 {v:26s} drift={d:.4e}")

    print(f"\n总耗时 {time.perf_counter()-t0:.2f} s")
    print("\n---- 建议 ----")
    hmin = min(r["headroom_min"] for r in all_rows)
    print(f"全部合法 kernel 的 headroom 最小值 = {hmin:.4f} "
          f"⇒ precision_slack_bits 建议取 {max(0.0, -hmin + 0.5):.2f} 左右")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
