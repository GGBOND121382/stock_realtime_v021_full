# Ch17 AS1455 现有模型回测与绘图

本分支当前只用于复用服务器上已经完成的模型检索结果，重新运行历史回测、strict-OOS forward 回测，并生成 fold6 至 fold0 的收益曲线。

## 安全边界

公开入口不会执行以下操作：

```text
更新行情缓存
重建 model_data
重新训练或重新检索模型
改名、删除或覆盖训练 fold 目录
```

以下模式已被主动禁止：

```text
all
preflight
data
selfcheck
training
historical
forward
plots
audit
status
```

这是为了防止将单纯回测误执行成完整重建。

## 唯一入口

```bash
bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
```

也可以直接调用：

```bash
bash scripts/run_ch17_as1455_backtest_only.sh
```

默认最低磁盘余量为 1 GiB：

```bash
MIN_FREE_GB=1 \
  bash scripts/run_ch17_as1455_full_rebuild.sh backtest-only
```

## 输入

脚本只读以下现有数据和模型：

```text
saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455_forward_latest/model_data_as1455.h5
saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache/
现有 fold0..fold6 / fold0..fold5 checkpoint、scaler 和 manifest
```

其中：

- `r01_fwd`、`r05_fwd` 使用已有 fold0..fold6；
- `r21_fwd` 使用已有 fold0..fold5；
- 脚本只检查这些产物是否存在，不修改训练目录；
- forward model_data 不存在时直接失败，不会自动重建。

## 执行内容

```text
读取现有 checkpoint
→ one-fold-lag 历史预测和回测
→ 从历史回测中选择完整最佳交易配置
→ 六组 fold0 strict-OOS forward 回测
→ 将六组 forward 起点统一到共同 fold0 边界后的首个可用交易日
→ 绘制 fold6、fold5、...、fold0 日/周/月收益曲线
```

历史映射保持原协议：

```text
source fold6 -> target fold5   # r1/r5 可用
source fold5 -> target fold4
...
source fold1 -> target fold0
```

`r21_fwd` 没有 source fold6，因此 fold6 图只包含 r1/r5 的 A/B 四条曲线；fold5 至 fold0 包含所有可用策略。

## 输出

每次运行使用独立时间戳。主要目录：

```text
saved_data/ashare_ml4t/ch17_as1455_target_backtest/
saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/
saved_data/ashare_ml4t/ch17_as1455_backtest_plots/backtest_only_<timestamp>/
saved_data/ashare_ml4t/ch17_as1455_backtest_only/<timestamp>/backtest_only_report.json
```

分 fold 图共 21 张：

```text
fold_sequence/fold6/return_curve_{daily,weekly,monthly}.png
...
fold_sequence/fold0/return_curve_{daily,weekly,monthly}.png
```

报告明确记录：

```text
data_refresh = false
model_data_rebuild = false
training = false
fold_plot_expected = 21
new_bytes_in_result_roots = 本次结果目录实际新增字节数
```
