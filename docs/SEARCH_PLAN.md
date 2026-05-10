# 分批搜索计划

## Phase P0：缓存矩阵

先将 `signal_samples.csv` 构造成可重复使用的 `.npz` 矩阵，避免每次重复构造标签和特征。

每个矩阵包含：

```text
X_train, X_valid, X_test
y_price_train/valid/test
y_delta_train/valid/test
current_vwap_valid/test
signal_time_valid/test
```

建议先准备：

```text
F1_start1000 + S2_core_lags
F3_1000_1430 + S2_core_lags
F4_0945_1430 + S2_core_lags
F5_0950_1430 + S2_core_lags
F6_0955_1430 + S2_core_lags
```

## Batch 0：baseline

跑所有过滤条件下的：

```text
pred = current_bar_vwap
pred = current_close
pred = session_vwap
pred_delta = 0
```

## Batch 1：线性 residual/delta

```text
Ridge alpha = 0.1, 1, 10, 100
BayesianRidge
ElasticNet
HuberRegressor
```

优先 F1/F3，目标 price 与 delta 都跑。

## Batch 2：ExtraTrees

已跑 F1 加深方向，无明显改善。剩余建议：换时间过滤，而不是继续加深。

```text
用当前最好 ET 参数跑：F3/F4/F5/F6
```

## Batch 3：LightGBM

优先 delta：

```text
num_leaves = 7, 15, 31
n_estimators = 40, 80, 120, 200
learning_rate = 0.03, 0.05, 0.08
min_child_samples = 30, 80, 120
```

分小批执行，每次一个配置，立即落盘。

## Batch 4：XGBoost

优先浅模型：

```text
max_depth = 2, 3
n_estimators = 80, 120, 200
learning_rate = 0.03, 0.05
min_child_weight = 20, 50
reg_lambda = 10, 30
```

## Batch 5：HistGradientBoosting

HGB 在当前对话环境很慢，但本地可以跑：

```text
max_iter = 40, 80, 160
max_leaf_nodes = 7, 15, 31
l2_regularization = 0.1, 1.0
```

## Batch 6：分时段模型

非常重要：

```text
S_morning: 10:00~11:30
S_afternoon: 13:00~14:30
```

每段单独训练 ExtraTrees / LightGBM / XGB。

## Batch 7：ensemble

取 valid_MAE 最好的 5~10 个模型：

```text
simple average
median ensemble
inverse valid_MAE weighted average
```

不得用 test 选择权重。

