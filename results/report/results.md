# 本地复现结果

以下均为独立实现的诊断实验，不能视为原论文图表复现。均值±样本标准差；错误越低越好。

| 数据 | 方法 | 层 | 完成种子数 | 未来追踪误差 mm | 相对本方法L0吞吐 |
|---|---|---:|---:|---:|---:|
| cloth | learned | 0 | 3 | 15.075 ± 12.264 | 1.00× |
| cloth | learned | 1 | 3 | 89.086 ± 26.029 | 1.46× |
| cloth | learned | 2 | 3 | 110.027 ± 41.541 | 1.55× |
| cloth | learned | 3 | 3 | 113.218 ± 28.436 | 1.59× |
| cloth | geometry | 0 | 3 | 11.611 ± 7.903 | 1.00× |
| cloth | geometry | 1 | 3 | 37.791 ± 3.973 | 1.22× |
| cloth | geometry | 2 | 3 | 76.566 ± 19.522 | 1.41× |
| cloth | geometry | 3 | 3 | 86.700 ± 2.104 | 1.52× |
| cloth | random | 0 | 3 | 6.900 ± 3.724 | 1.00× |
| cloth | random | 1 | 3 | 137.049 ± 8.761 | 1.46× |
| cloth | random | 2 | 3 | 178.758 ± 19.033 | 1.48× |
| cloth | random | 3 | 3 | 161.939 ± 8.967 | 1.65× |
| real_cloth_subset | learned | 0 | 3 | 20.837 ± 6.726 | 1.00× |
| real_cloth_subset | learned | 1 | 3 | 25.966 ± 0.514 | 1.63× |
| real_cloth_subset | learned | 2 | 3 | 28.807 ± 2.244 | 1.93× |
| real_cloth_subset | learned | 3 | 3 | 40.276 ± 11.884 | 2.01× |
| real_cloth_subset | geometry | 0 | 3 | 23.601 ± 6.183 | 1.00× |
| real_cloth_subset | geometry | 1 | 3 | 18.791 ± 2.489 | 1.65× |
| real_cloth_subset | geometry | 2 | 3 | 24.996 ± 4.383 | 1.90× |
| real_cloth_subset | geometry | 3 | 3 | 21.942 ± 2.598 | 2.01× |
| real_cloth_subset | random | 0 | 3 | 21.263 ± 5.623 | 1.00× |
| real_cloth_subset | random | 1 | 3 | 28.742 ± 6.155 | 1.54× |
| real_cloth_subset | random | 2 | 3 | 36.877 ± 7.707 | 1.84× |
| real_cloth_subset | random | 3 | 3 | 32.924 ± 3.363 | 2.30× |
| rope | learned | 0 | 3 | 35.785 ± 2.327 | 1.00× |
| rope | learned | 1 | 3 | 109.775 ± 6.409 | 1.29× |
| rope | learned | 2 | 3 | 119.901 ± 21.176 | 1.42× |
| rope | learned | 3 | 3 | 154.044 ± 19.638 | 1.42× |
| rope | geometry | 0 | 3 | 37.760 ± 4.767 | 1.00× |
| rope | geometry | 1 | 3 | 64.306 ± 18.250 | 1.30× |
| rope | geometry | 2 | 3 | 67.597 ± 15.922 | 1.41× |
| rope | geometry | 3 | 3 | 115.698 ± 12.101 | 1.28× |
| rope | random | 0 | 3 | 42.092 ± 11.028 | 1.00× |
| rope | random | 1 | 3 | 162.740 ± 28.898 | 1.27× |
| rope | random | 2 | 3 | 152.438 ± 18.304 | 1.41× |
| rope | random | 3 | 3 | 159.981 ± 9.383 | 1.47× |

原始16子步真实数据 learned/seed0 曾失败，见 `results/real_failure_audit/`；如有修复运行，表中明确使用 `real_repaired_0`，其余原始失败日志保留。

计时含投影/积分/轨迹恢复，不含渲染；多进程竞争影响CPU计时。不能与论文RTX4090端到端FPS直接比较。
