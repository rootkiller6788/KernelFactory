"""数值内核与候选空间的纯函数性质。

这些不是「再测一遍流水线」，而是流水线**依赖的前提**：

- `round_to_bits` 必须只做一次二进制舍入、且幂等；
- `effective_bits` 的 0 误差档必须是 `FULL_BITS`，不能是 inf；
- 分块/树形/顺序三种归约在边界上必须自洽；
- `enumerate_space` 的计数、剪枝理由、行算子常量基因必须与配置一致。

任何一条坏了，附录 B 的读数会整体偏移，而报告里**没有一个数会报错**。
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from kernelfactory.config import load_config
from kernelfactory.genome import (
    GENE_ORDER,
    KernelGenome,
    enumerate_space,
    gene_values,
    is_valid,
    space_digest,
)
from kernelfactory.numerics import (
    FULL_BITS,
    chunked_sum,
    effective_bits,
    expected_bits_bias,
    pair_sum,
    round_to_bits,
    scaled_max_error,
    seq_sum,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(root=ROOT)


# ------------------------------------------------------------------ 数值内核
def test_round_to_bits_is_identity_at_full_bits() -> None:
    x = 1.2345678901234567
    assert round_to_bits(x, FULL_BITS) == x


def test_round_to_bits_handles_special_values() -> None:
    assert round_to_bits(0.0, 8) == 0.0
    assert math.isnan(round_to_bits(float("nan"), 8))
    assert round_to_bits(float("inf"), 8) == float("inf")
    with pytest.raises(ValueError):
        round_to_bits(1.0, 0)


def test_round_to_bits_is_idempotent() -> None:
    for x in (0.1, 1.0, 3.1415926535, -2.718281828):
        once = round_to_bits(x, 8)
        assert round_to_bits(once, 8) == once


def test_effective_bits_endpoints() -> None:
    assert effective_bits(0.0) == float(FULL_BITS)
    assert effective_bits(2.0 ** -8) == pytest.approx(8.0)


def test_expected_bits_bias_endpoints() -> None:
    assert expected_bits_bias(0) == 0.0
    assert expected_bits_bias(1) == 0.0
    assert expected_bits_bias(16) == pytest.approx(2.0)


def test_sum_families_agree_on_exact_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert pair_sum([], FULL_BITS) == 0.0
    assert pair_sum(values, FULL_BITS) == 10.0
    assert seq_sum(values, FULL_BITS) == 10.0
    # tile_k 覆盖整个序列 ⇒ 退化成树形归约（同一条代码路径）。
    assert chunked_sum(values, len(values), FULL_BITS) == pair_sum(values, FULL_BITS)


def test_scaled_max_error_is_normalized_by_problem_scale() -> None:
    assert scaled_max_error([0.0, 0.0], [0.0, 0.0], 0.0) == 0.0
    # 分母是 max|want| = 1，不是逐元素的 |want_i|（后者会让近零元素爆炸）。
    assert scaled_max_error([1.0, 2.0], [1.0, 1.0], 0.0) == pytest.approx(1.0)
    assert scaled_max_error([0.0], [10.0], 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        scaled_max_error([1.0], [1.0, 2.0], 0.0)


# ------------------------------------------------------------------ 候选空间
def test_genome_roundtrip_and_identity() -> None:
    g = KernelGenome.from_dict(load_config(root=ROOT)["baseline"]["matmul"])
    assert g.to_dict() == {k: getattr(g, k) for k in GENE_ORDER}
    assert set(g.to_dict()) == set(GENE_ORDER)
    assert KernelGenome.from_dict(g.to_dict()) == g
    assert g.digest() == KernelGenome.from_dict(g.to_dict()).digest()
    assert g.digest() != dataclasses.replace(g, accum_dtype="bf16").digest()
    assert g.label().startswith(g.accum_dtype + "/")


def test_from_dict_rejects_missing_gene() -> None:
    d = dict(load_config(root=ROOT)["baseline"]["matmul"])
    d.pop("tile_m")
    with pytest.raises(KeyError):
        KernelGenome.from_dict(d)


def test_gene_values_sorts_numbers_but_keeps_string_order() -> None:
    vals = gene_values(load_config(root=ROOT))
    assert list(vals["tile_m"]) == sorted(vals["tile_m"])
    assert list(vals["memory_layout"]) == ["naive", "swizzled", "vectorized"]


def test_is_valid_returns_reasons() -> None:
    cfg = load_config(root=ROOT)
    base = KernelGenome.from_dict(cfg["baseline"]["matmul"])

    def why(**changes: object) -> str:
        return is_valid(dataclasses.replace(base, **changes), cfg, "matmul")[1]

    assert why(num_stages=2, scheduling_strategy="default") == "stages_without_pipelining"
    assert why(memory_layout="vectorized", vector_width=1) == "vectorized_without_vector"
    assert why(tile_m=1, tile_n=8, num_warps=8) == "too_many_warps_for_tile"
    assert why() == "ok"


def test_space_counts_and_prune_reasons(cfg: dict) -> None:
    counts: dict[str, int] = {}
    for op in ("layernorm", "matmul", "softmax"):
        space, diag = enumerate_space(cfg, op)
        counts[op] = len(space)
        assert len(space) == diag["n_distinct"]
        assert len({g.digest() for g in space}) == len(space)
        assert [g.digest() for g in space] == sorted(g.digest() for g in space)
        assert diag["n_pruned"] == sum(diag["prune_reasons"].values())
    assert counts == {"layernorm": 450, "matmul": 5400, "softmax": 450}


def test_row_ops_force_constant_genes(cfg: dict) -> None:
    space, diag = enumerate_space(cfg, "layernorm")
    assert diag["gene_cardinality"]["tile_m"] == 1
    assert diag["gene_cardinality"]["tile_k"] == 1
    assert all(g.tile_m == 1 and g.tile_k == 1 for g in space)


def test_space_digest_is_order_independent(cfg: dict) -> None:
    space, _ = enumerate_space(cfg, "layernorm")
    assert space_digest(space) == space_digest(list(reversed(space)))
    again, _ = enumerate_space(cfg, "layernorm")
    assert space_digest(space) == space_digest(again)
