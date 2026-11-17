"""KernelFactory：Kernel 自动生成的搜索闭环（硬约束 = 数值正确性）。

形状：**搜索闭环 + 硬约束 · 第三形态（数值容差）**。
前两形态是 `L5-modeloptimizationfactory`（资源轴）与
`L6-graphoptimizationfactory`（语义等价）。

分水岭：**约束在这里真的 binding** —— 放宽容差真的买到速度，
而 graphopt 里放宽容差买到的质量是零。详见 `report.py` 的头条。
"""

from __future__ import annotations

__version__ = "1.0.0"
