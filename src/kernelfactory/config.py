"""配置：`DEFAULTS` + 深合并 + 加载 + 校验 + 协议指纹。

**所有阈值 / 权重 / 容差 / 基因取值表都在这里，代码里不硬编码**（工程约定）。
`configs/default.json` 是 `DEFAULTS` 的落盘副本，`dump-config` 会重新生成它。

⚠️ 深合并里**列表是整体替换、磁盘优先** —— 于是「改了代码里的声明、重跑，
数字一字未变」是一个真实踩过的坑（附录 C 204）。所以有一条保险丝
`no_stale_config_keys`：磁盘上多出来的键会让流水线报错，而不是被静默忽略。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .genome import GENE_ORDER, ROW_CONSTANT_GENES
from .numerics import DTYPE_SPEEDUP, SIGNIFICANT_BITS

DEFAULTS: dict[str, Any] = {
    "seed": 20261004,

    # ---- 算子：探测形状（数值孪生）与产出形状（成本模型）**刻意不同** ----
    #
    # ⚠️ 两个尺度不一样是**刻意的**，也是必须写明的：本机没有 GPU，
    #    在 512³ 上真的跑一遍 Python 浮点要几分钟。所以延迟是**解析模型**
    #    在产出形状上算的，误差是**真跑**在探测形状上量的。
    #    把假设标成测量是这类报告最危险的失真，所以两个形状都进产物。
    #
    # ⚠️ 探测形状还必须**让基因取值表整体可达**：
    #    `genome.tile_n` 最大是 32，而过滤规则要求 `tile_n <= N`。
    #    第一版 matmul 取 N=24 ⇒ `tile_n=32` **被整体砍掉** ——
    #    于是「我们在搜 tile_n 这个基因」是一句假话，
    #    而空间大小看起来只是「小了一点」（3600 → 2400）。
    #    有一条对抗用例 `every_gene_value_survives_filtering` 专门抓这个。
    "ops": {
        "matmul": {
            "row_wise": False,
            "probe": [24, 32, 32],
            "production": [512, 512, 512],
            "input_scale": 0.6,
        },
        "softmax": {
            "row_wise": True,
            "probe": [16, 48, 1],
            "production": [256, 4096, 1],
            "input_scale": 2.0,
        },
        "layernorm": {
            "row_wise": True,
            "probe": [16, 48, 1],
            "production": [256, 4096, 1],
            "input_scale": 1.5,
            #: LayerNorm 的数值稳定项。**它是算子定义的一部分，不是容差** ——
            #: 非法变体 `eps_zero` 就是把它偷偷设成 0。
            "eps": 1e-5,
        },
    },

    # ---- KernelGenome 的取值表（9 个基因；README 原稿 8 个 + accum_dtype）----
    #
    # 为什么加第 9 个基因：**没有精度轴，`Error(K) < ε` 在基因空间上就是常数**，
    # 那条硬约束要么永远成立要么永远不成立 —— 它不构成搜索问题。
    # 详见 `genome.py` 的模块文档。
    "genome": {
        "tile_m": [8, 16],
        "tile_n": [8, 16, 32],
        "tile_k": [8, 16, 32],
        "num_warps": [1, 2, 4, 8],
        "num_stages": [1, 2, 3],
        "memory_layout": ["naive", "swizzled", "vectorized"],
        "vector_width": [1, 2],
        "scheduling_strategy": ["default", "pipelined", "persistent"],
        "accum_dtype": ["fp32", "tf32", "bf16"],
    },

    # ---- 手写基线：**所有 drift 的比较基准** ----
    #
    # 刻意取空间里最保守的一档（naive / 标量 / 1 warp / 单 stage / fp32），
    # 于是有充足的优化余量。基线自己的 drift 恰好是 0。
    # ⚠️ 分块必须**不整除**探测形状**，否则尾部类非法变体恒等于 0。
    #    M=24 / tile_m=16 ⇒ 尾巴 8 行；N=48 / tile_n=32 ⇒ 尾巴 16 列。
    #    第一版取 tile_m=8 / tile_n=8（恰好整除），于是 `drop_tail` 的
    #    漂移**精确等于 0** —— 它看起来像「这个变体没生效」，
    #    其实是用例根本没被执行到。
    "baseline": {
        "matmul": {"tile_m": 16, "tile_n": 16, "tile_k": 8, "num_warps": 1,
                   "num_stages": 1, "memory_layout": "naive", "vector_width": 1,
                   "scheduling_strategy": "default", "accum_dtype": "fp32"},
        "softmax": {"tile_m": 1, "tile_n": 32, "tile_k": 1, "num_warps": 1,
                    "num_stages": 1, "memory_layout": "naive", "vector_width": 1,
                    "scheduling_strategy": "default", "accum_dtype": "fp32"},
        "layernorm": {"tile_m": 1, "tile_n": 32, "tile_k": 1, "num_warps": 1,
                      "num_stages": 1, "memory_layout": "naive", "vector_width": 1,
                      "scheduling_strategy": "default", "accum_dtype": "fp32"},
    },

    # ---- 数值孪生的两个数值参数 ----
    "numerics": {
        #: 相对误差分母的下限。**它不是容差**，是防除零的量纲垫片：
        #: 没有它，一个真值为 0 的输出元素会把误差放大成 inf，
        #: 而那个 inf 会一路传进「最大误差」把整个读数变成废数。
        "error_floor": 1e-12,
        #: 门 2 的 slack（位）。**它是标定出来的，不是拍的**：
        #: `bits_headroom = 有效位宽 − (声明位宽 − √n 偏置)`，
        #: slack 取「全部合法 kernel 的 headroom 最小值」再留一点余量。
        #: 标定脚本在 `.scratch/calibrate.py`，结果印在 `numerics.json`。
        "precision_slack_bits": 0.6,
    },

    # ---- 硬约束：一致性容差 τ ----
    "semantics": {
        #: **「未测量时会选的容差」**，不是推荐值。
        #: 本项目的一条结论就是它取错了：1e-5 是 `torch.allclose` 的常见默认，
        #: 而它会**拒掉全部 tf32 与 bf16** —— 也就是拒掉这个项目全部的加速来源。
        "tolerance": 1e-5,
    },

    # ---- 搜索 ----
    "search": {
        "budget_per_searcher": 120,
        #: 紧预算：用来**区分策略**的预算。
        "tight_budget": 20,
        "beam_width": 4,
        "population": 12,
        "mutation_rate": 0.35,
        "searchers": ["random", "greedy", "beam", "evolution"],
        "reference_pad": 0.02,
    },

    # ---- 约束处理策略 ----
    "constraints": {
        "strategy": "reject",          # reject | penalty
        "penalty_weight": 1.0,
        #: 违反量的**固定**分母（`violation = max(0, drift−τ)/drift_scale`）。
        #: **不是「除以容差」** —— 除以 τ 会让惩罚在 τ→0 时退化成硬约束，
        #: 于是「软约束不保证可行」这条结论永远暴露不出来。
        #: 取 1.0 = 「漂移本身」就是量纲：逐位一致的 kernel 违反量 ~1e-7（几乎不罚），
        #: 一个算错的 kernel 违反量 ~1.0（重罚）。
        "drift_scale": 1.0,
    },

    # ---- 阶梯：逐级放松**一条**约束（一致性容差 τ）----
    "ladder": {
        #: **分辨率放在信号所在的地方。** 三个精度档的漂移量级是
        #: fp32 ~1e-7 / tf32 ~1e-3 / bf16 ~1e-2，所以底部四级
        #: （0 / 1e-8 / 1e-7 / 1e-6）分辨 fp32 内部的累加顺序差异，
        #: 中间四级跨过 tf32，顶部两级跨过 bf16。
        #: 标定过程见 `.scratch/calibrate.py` —— **先量误差，再定阶梯**。
        "tolerances": [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },

    # ---- 预算扫描：单一预算点不是比较，是轶事 ----
    "budget_sweep": {
        "budgets": [10, 20, 30, 40, 60, 80, 120],
        "strategy": "reject",
        "tolerance": 1e-3,
    },

    # ---- 对抗：注入的**已知幅度**扰动（尺子）----
    "adversarial": {
        #: 尺子：`mantissa_truncate(b)` —— 声称 fp32，实际按 b 位有效跑。
        #: 漂移 ≈ 2^−b，于是「崖在哪」是一个**可解析定位**的数。
        #: ⚠️ b 网格必须横跨崖的两侧（否则崖没被测量到，而表看起来很正常）。
        "mantissa_bits": [2, 4, 6, 8, 10, 12, 16, 20],
    },

    # ---- 成本模型常数：**全部是假设**，不是测量 ----
    "cost": {
        "launch_overhead_us": 3.0,
        #: 20 TFLOPs/µs = 2e7 flops/µs —— 消费级 GPU 的 fp32 峰值量级。
        "flops_per_us": 20000000.0,
        #: 400 GB/s = 4e5 bytes/µs —— 显存带宽量级（保守取有效值）。
        "bytes_per_us": 400000.0,
        #: 分块算术强度的**饱和点**（flops/byte）：低于它说明这个分块
        #: 榨不出全部并行度。**它是假设**；机器平衡点（flops_per_us÷bytes_per_us = 50）
        #: 量的是另一件事（「问题算不算访存瓶颈」），两者不能混用。
        "ai_saturate": 8.0,
        #: 各精度**输入**的字节数。tf32 仍是 4 字节 —— 它在显存里就是 fp32，
        #: 只有乘法器截断。**这是 tf32 与 bf16 的物理差别**：
        #: tf32 只买算力，bf16 同时买带宽。
        "dtype_bytes_per_elem": {"fp32": 4, "tf32": 4, "bf16": 2},
        "row_op_flops_per_elem": 8.0,
        "layout_eff": {"naive": 0.80, "swizzled": 1.00, "vectorized": 1.10},
        "vec_eff": {"1": 0.85, "2": 1.00},
        "sched_eff": {"default": 1.00, "pipelined": 1.10, "persistent": 1.20},
    },
    "cost_sweep": {
        "launch_overhead_us": [0.0, 0.3, 1.0, 3.0, 10.0, 30.0],
    },

    # ---- 验收门槛：**每一条都必须能红** ----
    #
    # 写门槛时问两个问题：① 它在什么情况下会红？② 那个情况是真的错，还是运气不好？
    # 答不上来的那一条删掉 —— 一条恒真的判据与没有判据在输出里长得一模一样。
    "acceptance": {
        # -- 结构性：三个算子都要有空间，且空间不能全是同一个样子
        "min_ops": 3,
        "min_space_per_op": 200,
        "min_distinct_objective_vectors": 100,
        "max_structural_failures": 0,
        # -- 精度轴必须真的在基因里（否则约束是常数）
        "min_distinct_accum_dtypes": 3,
        "min_distinct_effective_bits": 3,
        # -- 硬约束必须**真的 binding**：放宽容差要能买到速度
        #: 这是本项目与 L6-graphopt 的分水岭。那边放宽容差买到的质量是**零**
        #: （前沿一格没动）；这边必须买得到，否则说明精度轴没接进成本模型。
        "min_ladder_speedup_gain": 1.0,
        # -- 两层门：门 2 必须真的拒掉东西，否则它是装饰
        "min_precision_gate_rejections": 1,
        # -- 前沿
        "min_frontier_points": 2,
        "max_greedy_gap_vs_oracle": 0.35,
        "oracle_is_optimal": True,
        # -- 交付判据：交出去的答案必须可行**且诚实**
        "max_answer_infeasible_rate": 0.0,
        "max_answer_dishonest_rate": 0.0,
        # -- 阶梯单调（数学必然，红了就是实现错了）
        "min_monotone_strategies": 2,
        # -- 对抗用例（**不可调的不变量断言**）不许被偷偷删掉
        "min_adversarial_cases": 30,
        # -- 保险丝：验收表本身不许被偷偷缩短
        "min_acceptance_checks": 18,
    },

    # ---- 设计期望（**允许被实测推翻**，被推翻的那条最值钱）----
    "expectations": {
        "three_objectives_are_independent": {
            "claim": "三个目标轴互不冗余（两两 |corr| < 0.95）",
            "expect": True,
        },
        "tile_k_trades_accuracy_for_speed": {
            "claim": "tile_k 越大越准（块内规约树更深）但越慢",
            "expect": True,
        },
        "num_warps_is_an_accuracy_gene": {
            "claim": "num_warps 影响误差（块内多一层规约树）—— 它不是纯速度基因",
            "expect": True,
        },
        "memory_layout_is_numerically_inert": {
            "claim": "memory_layout 对误差的影响**恰好为零**（它只改访存，不改加法树）",
            "expect": True,
        },
        "ladder_buys_speed": {
            "claim": "放宽容差能买到速度（最优可行 kernel 的加速比随 τ 上升）",
            "expect": True,
        },
        "safe_tolerance_window_is_nonempty": {
            "claim": "存在一个容差区间：全部**合法**精度通过、全部**非法**扰动被拒",
            "expect": True,
        },
        "precision_gate_is_needed": {
            "claim": "存在一个 τ：门 1 放行、门 2 拒掉的 kernel 真的存在",
            "expect": True,
        },
        "effective_bits_has_a_sqrt_n_bias": {
            "claim": "实测有效位宽 ≈ 声明位宽 − 0.5·log2(累加步数)（在 1 位以内）",
            "expect": True,
        },
        "bf16_is_free": {
            "claim": "bf16 在宽松容差下既最快又仍然诚实（门 2 不拒它）",
            "expect": True,
        },
        "default_tolerance_admits_all_precisions": {
            "claim": "`semantics.tolerance` 的默认值能收下全部三个精度档",
            "expect": True,
        },
        "searchers_beat_random": {
            "claim": "在主预算下，非随机搜索器都优于随机搜索",
            "expect": True,
        },
        "random_saturates_at_main_budget": {
            "claim": "主预算下随机搜索已拿到 oracle 的全部超体积",
            "expect": True,
        },
        "soft_constraint_changes_the_answer": {
            "claim": "把硬约束换成软约束会改变交出去的答案",
            "expect": True,
        },
        "launch_overhead_dominates_the_speedup": {
            "claim": "把 launch_overhead_us 扫到 0 后，最优 kernel 的加速比大幅缩小",
            "expect": True,
        },
        "analytic_latency_tracks_wallclock": {
            "claim": "解析延迟与实测求值时间的排序同向（秩相关 ≥ 0.5）",
            "expect": True,
        },
        "precision_gate_never_rejects_legal": {
            "claim": "门 2 不误伤任何合法 kernel（slack 标定得够宽）",
            "expect": True,
        },
    },
}


class ConfigError(Exception):
    """配置不合法。**一律抛错**，不静默回落（附录 C 2/3：宽松的 load_config
    会把「路径错了」伪装成「用默认值」）。"""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并；**列表整体替换**（这是坑 204 的来源，所以有 `no_stale_config_keys`）。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, *,
                root: str | Path | None = None) -> dict[str, Any]:
    """加载配置。

    `root` **真的参与路径解析** —— 判据不是「调用点写没写 `root=`」，
    而是「这个参数有没有被用上」（附录 C 判据）。
    """
    if path is None:
        base = Path(root) if root is not None else Path.cwd()
        p = base / "configs" / "default.json"
    else:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        disk = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件不是合法 JSON：{p} —— {e}") from e
    cfg = deep_merge(DEFAULTS, disk)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(Path(root) if root is not None else Path.cwd())

    # ⚠️ 这条保险丝**必须真的被调用**。写好了却不接在这里，后果是：
    #    从 `DEFAULTS` 里删掉的键会继续活在磁盘配置里，被 `deep_merge` 带进
    #    运行配置，于是验收表里多出一条「没有任何检查函数认领」的门槛 ——
    #    它永远是红的，而真因（配置陈旧）完全看不出来。
    #    **一个写了但没人调的检查，与没有这个检查在输出里完全一样。**
    stale = no_stale_config_keys(disk, DEFAULTS)
    if stale:
        raise ConfigError(
            "配置里有代码中不存在的键（多半是从 DEFAULTS 删了键但磁盘配置没跟着更新）：\n"
            + "\n".join(f"  - {k}" for k in stale)
            + f"\n  配置文件：{p}\n"
            "  修法：用 `kernelfactory.config.dump_config(DEFAULTS, path)` 重新导出。"
        )
    validate_config(cfg)
    return cfg


def no_stale_config_keys(disk: dict[str, Any], defaults: dict[str, Any],
                         prefix: str = "") -> list[str]:
    """磁盘上出现了代码里没有的键 ⇒ 报出来（保险丝 `no_stale_config_keys`）。"""
    stale: list[str] = []
    for k, v in disk.items():
        here = f"{prefix}{k}"
        if k not in defaults:
            stale.append(here)
        elif isinstance(v, dict) and isinstance(defaults[k], dict):
            stale.extend(no_stale_config_keys(v, defaults[k], here + "."))
    return stale


def validate_config(cfg: dict[str, Any]) -> None:
    """配置校验。**每条报错写清「为什么这是错的」** —— 文案本身是被断言的一部分。"""
    errs: list[str] = []

    if int(cfg["seed"]) <= 0:
        errs.append("seed 必须 > 0（0 不是种子，是「没设种子」）")

    # ---- 算子 ----
    if not cfg["ops"]:
        errs.append("ops 不能为空 —— 一个算子的流水线不叫「多算子」")
    for name, d in cfg["ops"].items():
        p = [int(x) for x in d["probe"]]
        q = [int(x) for x in d["production"]]
        if len(p) != 3 or len(q) != 3:
            errs.append(f"ops.{name}.probe / production 必须是三元组 (M, N, K)")
            continue
        if min(p) < 1 or min(q) < 1:
            errs.append(f"ops.{name} 的形状分量必须 ≥ 1")
        if bool(d["row_wise"]) and (p[0] < 1 or p[1] < 2):
            errs.append(f"ops.{name} 是行式算子，探测形状至少要有 2 列")
        if bool(d["row_wise"]) and p[2] != 1:
            errs.append(f"ops.{name} 是行式算子，K 维必须为 1"
                        "（行式算子在 K 方向没有可切的维度）")
        if float(d["input_scale"]) <= 0:
            errs.append(f"ops.{name}.input_scale 必须 > 0")
    if "layernorm" in cfg["ops"] and float(cfg["ops"]["layernorm"]["eps"]) <= 0:
        errs.append("ops.layernorm.eps 必须 > 0 —— 它是算子定义的一部分，"
                    "取 0 就是那个非法变体 `eps_zero`")

    # ---- 基因 ----
    g = cfg["genome"]
    for name in GENE_ORDER:
        if name not in g:
            errs.append(f"genome.{name} 缺失 —— 9 个基因一个都不能少")
        elif not g[name]:
            errs.append(f"genome.{name} 的取值表为空 —— 那个基因会从空间里消失")
    for name in ("tile_m", "tile_n", "tile_k", "num_warps", "num_stages",
                 "vector_width"):
        if name in g and any(int(v) < 1 for v in g[name]):
            errs.append(f"genome.{name} 必须全部 ≥ 1")
    if "accum_dtype" in g:
        unknown = [x for x in g["accum_dtype"] if x not in SIGNIFICANT_BITS]
        if unknown:
            errs.append(f"genome.accum_dtype 含未知精度 {unknown}；"
                        f"已知：{sorted(SIGNIFICANT_BITS)}")
        if len(set(g["accum_dtype"])) < 2:
            errs.append("genome.accum_dtype 至少要有 2 档 —— **只有一档等于没有精度轴**，"
                        "而 `Error(K) < ε` 在基因空间上会变成常数")
    if "memory_layout" in g:
        unknown = [x for x in g["memory_layout"] if x not in cfg["cost"]["layout_eff"]]
        if unknown:
            errs.append(f"genome.memory_layout 含成本模型里没有的布局 {unknown}")
    if "scheduling_strategy" in g:
        unknown = [x for x in g["scheduling_strategy"]
                   if x not in cfg["cost"]["sched_eff"]]
        if unknown:
            errs.append(f"genome.scheduling_strategy 含成本模型里没有的调度 {unknown}")
    if "vector_width" in g:
        unknown = [str(x) for x in g["vector_width"] if str(x) not in cfg["cost"]["vec_eff"]]
        if unknown:
            errs.append(f"genome.vector_width 含成本模型里没有的宽度 {unknown}")

    # ---- 基线 ----
    for op in cfg["ops"]:
        if op not in cfg["baseline"]:
            errs.append(f"baseline.{op} 缺失 —— 每个算子都要有一个手写基线，"
                        "它是所有 drift 的比较基准")
            continue
        b = cfg["baseline"][op]
        missing = [n for n in GENE_ORDER if n not in b]
        if missing:
            errs.append(f"baseline.{op} 缺字段 {missing}")
            continue
        if bool(cfg["ops"][op]["row_wise"]):
            for n in ROW_CONSTANT_GENES:
                if int(b[n]) != 1:
                    errs.append(f"baseline.{op}.{n} 必须是 1 —— "
                                f"{n} 在行式算子上不参与搜索")

    # ---- 数值 ----
    if float(cfg["numerics"]["error_floor"]) <= 0:
        errs.append("numerics.error_floor 必须 > 0 —— 它是防除零的垫片，"
                    "取 0 会让真值为 0 的元素把误差放大成 inf")
    if float(cfg["numerics"]["precision_slack_bits"]) < 0:
        errs.append("numerics.precision_slack_bits 不能为负 —— 负 slack 会让门 2 "
                    "比声明本身还严，把合法 kernel 也一起拒掉")

    if float(cfg["semantics"]["tolerance"]) < 0:
        errs.append("semantics.tolerance 不能为负（负容差会让任何 kernel 都不可行）")

    # ---- 搜索 ----
    s = cfg["search"]
    for k in ("budget_per_searcher", "tight_budget", "beam_width", "population"):
        if int(s[k]) < 1:
            errs.append(f"search.{k} 必须 ≥ 1")
    if int(s["population"]) < 2:
        errs.append("search.population 必须 ≥ 2（进化搜索至少要两个个体才能交叉）")
    if int(s["tight_budget"]) >= int(s["budget_per_searcher"]):
        errs.append("search.tight_budget 必须 < budget_per_searcher"
                    "（否则「紧预算」就是主预算，两个读数会一模一样）")

    if cfg["constraints"]["strategy"] not in ("reject", "penalty"):
        errs.append("constraints.strategy 只能是 reject / penalty")
    if float(cfg["constraints"]["penalty_weight"]) < 0:
        errs.append("constraints.penalty_weight 不能为负")

    # ---- 阶梯 ----
    tol = [float(x) for x in cfg["ladder"]["tolerances"]]
    if len(tol) < 3:
        errs.append("ladder.tolerances 至少 3 级（少了看不出阶梯）")
    if tol != sorted(tol):
        errs.append("ladder.tolerances 必须单调不减（阶梯的方向就是放松约束）")
    if 0.0 not in tol:
        errs.append("ladder.tolerances 必须含 0.0 —— 那是「只接受逐位一致」那一级")
    if len(set(tol)) != len(tol):
        errs.append("ladder.tolerances 有重复档 —— 重复档会让阶梯里出现两行"
                    "完全一样的读数，而它们看起来像两个独立的证据")

    # ---- 对抗 ----
    mb = [int(x) for x in cfg["adversarial"]["mantissa_bits"]]
    if len(mb) < 3:
        errs.append("adversarial.mantissa_bits 至少 3 档（少了定位不出崖）")
    if mb != sorted(mb):
        errs.append("adversarial.mantissa_bits 必须严格递增"
                    "（乱序的扫描表会印出一张假的单调表）")
    if len(set(mb)) != len(mb):
        errs.append("adversarial.mantissa_bits 有重复档")
    if max(mb) >= 53:
        errs.append("adversarial.mantissa_bits 必须 < 53（53 位就是「没有截断」，"
                    "那不是一条非法变体）")
    # ⚠️ **这里刻意不断言「网格必须横跨崖的两侧」。**
    # 是否被接受取决于**实测漂移**，而配置校验只看得到位数与容差两个数、
    # 量不出漂移。在只能看见代理量的地方断言代理量之间的关系，
    # 等于在配置里硬编码一个没人测量过的系数 —— 改一次形状，这条校验就变成
    # 一句假话，而且它永远是绿的。真正的断言写在 `adversarial.py`。
    # **校验只该断言它量得到的东西。**

    # ---- 成本 ----
    c = cfg["cost"]
    for k in ("launch_overhead_us", "flops_per_us", "bytes_per_us",
              "ai_saturate", "row_op_flops_per_elem"):
        if float(c[k]) <= 0:
            errs.append(f"cost.{k} 必须 > 0（0 会让对应那一维从成本里消失）")
    for dt_ in SIGNIFICANT_BITS:
        if dt_ not in c["dtype_bytes_per_elem"]:
            errs.append(f"cost.dtype_bytes_per_elem 缺 {dt_} —— "
                        f"那个精度档的访存会被静默按 0 字节算，"
                        f"于是它看起来最快")
        elif float(c["dtype_bytes_per_elem"][dt_]) <= 0:
            errs.append(f"cost.dtype_bytes_per_elem.{dt_} 必须 > 0")
    for key in ("layout_eff", "sched_eff"):
        for name, v in c[key].items():
            if float(v) <= 0:
                errs.append(f"cost.{key}.{name} 必须 > 0")
    for name, v in DTYPE_SPEEDUP.items():
        if name not in SIGNIFICANT_BITS:
            errs.append(f"numerics.DTYPE_SPEEDUP 里有未知精度 {name}")

    # ---- 验收 ----
    acc = cfg["acceptance"]
    if int(acc["min_ops"]) < 2:
        errs.append("acceptance.min_ops 必须 ≥ 2（README 的验收标准要求 ≥ 3 类算子）")
    if int(acc["min_space_per_op"]) < 10:
        errs.append("acceptance.min_space_per_op 必须 ≥ 10（空间太小不是搜索，是枚举）")
    if float(acc["max_answer_infeasible_rate"]) != 0.0:
        errs.append("acceptance.max_answer_infeasible_rate 必须是 0.0"
                    "—— 交出去的答案必须可行，这是硬约束的定义")
    if float(acc["max_answer_dishonest_rate"]) != 0.0:
        errs.append("acceptance.max_answer_dishonest_rate 必须是 0.0"
                    "—— 交出去的答案必须诚实，这是**门 2** 的定义")
    if float(acc["min_ladder_speedup_gain"]) <= 0:
        errs.append("acceptance.min_ladder_speedup_gain 必须 > 0"
                    "—— 取 0 等于声明「放宽容差买不到速度」，"
                    "那与 L6-graphopt 的结论就分不开了")
    if not bool(acc["oracle_is_optimal"]):
        errs.append("acceptance.oracle_is_optimal 必须为 True"
                    "—— oracle 枚举了整个空间，它不是最优就说明枚举或评估有 bug")
    if int(acc["min_acceptance_checks"]) < 10:
        errs.append("acceptance.min_acceptance_checks 必须 ≥ 10"
                    "（这条保险丝自己也不许被调松）")

    for name, spec in cfg["expectations"].items():
        if not isinstance(spec, dict) or "claim" not in spec or "expect" not in spec:
            errs.append(f"expectations.{name} 必须有 claim 与 expect 两个字段")

    if errs:
        raise ConfigError("配置不合法：\n  - " + "\n  - ".join(errs))


def protocol_digest(cfg: dict[str, Any]) -> str:
    """协议指纹：**只覆盖影响结论的字段**，不含路径。"""
    from .rng import digest_of
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return digest_of(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    """把当前生效的配置落盘（`DEFAULTS` 的副本 + 覆盖）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
