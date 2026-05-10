# 当前已完成搜索结果摘要

## 目标

预测未来 6 根 5 分钟 `bar_vwap` 的平均值：

```text
target_price = mean(bar_vwap[t+1:t+6])
delta = target_price - current_bar_vwap
```

评估统一还原到 `target_price` 后计算绝对误差。

## 已完成结果

| 批次 | 配置 | test_MAE | test_p90_AE | test_max_AE | <=0.05比例 |
|---|---|---:|---:|---:|---:|
| Batch 0 | baseline current_close, F1_start1000 | ~0.08823 | ~0.19778 | ~1.29829 | ~43.55% |
| Batch 2A | ExtraTrees d8 leaf20, delta, F1 | ~0.08836 | ~0.19794 | ~1.27202 | ~42.92% |
| Batch 2B-1 | ExtraTrees d10 leaf10, n80, delta, F1 | 0.08842 | 0.19901 | 1.27029 | 42.81% |
| Batch 2B-2 | ExtraTrees d12 leaf10, n80, delta, F1 | 0.08856 | 0.19881 | 1.24615 | 42.98% |
| Batch 2B-3 | ExtraTrees d14 leaf5, n80, delta, F1 | 0.08861 | 0.19883 | 1.24627 | 42.75% |
| Batch 3A-1-small | LightGBM leaves7 n20, delta, F1 | 0.08844 | 0.19817 | 1.26614 | 43.53% |
| Batch 7A | Ridge/BayesianRidge delta, F1 | ~0.08875 | ~0.19876 | ~1.29522 | ~42.43% |

## 注意

1. 这些结果尚不足以证明 `0.05` 不可能达到，只说明当前已跑配置没有达到。
2. ExtraTrees 加深没有改善，继续加深优先级应降低。
3. LightGBM 小配置能跑，但未改善；可继续跑更系统的 LightGBM/XGB/分时段模型。
4. 下一步应重点跑：时间过滤 F3/F4/F5/F6、分时段模型、LightGBM/XGB 小网格、ensemble。

