# AS1455 checkpoint-compatible live signal v1

本补丁适配当前 `scripts/run_ashare_ch17_nn_reproduce.py` 的真实训练产物：

```text
results/best_params.csv
results/cv_split_report.csv
results/logs/{dense_layers}/{activation}/{dropout}/{batch_size}/ckpt_{fold}_{epoch}.weights.h5
```

它不再假设存在 `model_0.keras`。第一版默认使用现有 checkpoint，不重训。

## 安装

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_checkpoint_signal_onekey.zip
bash as1455_checkpoint_signal_onekey/install.sh --repo .
```

## 默认配置

```bash
TRAIN_RUN_DIR=saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7
MODEL_DATA=saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5
DEPLOY_DIR=saved_data/ashare_ml4t/ch17_as1455_deploy/sharpe1_checkpoint_ensemble_all5_v1
MODEL_ROWS=0,1,2,3,4
FOLDS=0,1,2,3,4,5,6
FOLD_MODE=mean_all_folds
```

其中 `MODEL_ROWS=0,1,2,3,4` 对应 `best_params.csv` 前五行，也就是回测里的 `model_0..model_4`。

## 只创建/检查 checkpoint bundle

```bash
python3 tools/create_as1455_sharpe1_checkpoint_bundle_v1.py \
  --train-run-dir saved_data/ashare_ml4t/ch17_as1455_train_20260622_cv7 \
  --out-dir saved_data/ashare_ml4t/ch17_as1455_deploy/sharpe1_checkpoint_ensemble_all5_v1 \
  --model-data saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5 \
  --model-rows 0,1,2,3,4 \
  --folds 0,1,2,3,4,5,6 \
  --fold-mode mean_all_folds \
  --force
```

## 已有 live 特征后，只跑 checkpoint 推理/rank/signal

前提：当天目录里已经有：

```text
11_live_model_features_for_prediction.csv
```

运行：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

回放某天：

```bash
TRADE_DATE=20260626 FORCE_REBALANCE=1 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

干跑检查 checkpoint/scaler，不加载 TensorFlow 权重预测：

```bash
TRADE_DATE=20260626 DRY_RUN=1 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

## 一键：fastpath + checkpoint signal

如果已经安装 `run_as1455_live_fast_auto_v1.sh`：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_checkpoint_signal_v1.sh auto
```

14:50 后：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_checkpoint_signal_v1.sh post
```

只跑信号层：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_checkpoint_signal_v1.sh signal
```

## 可指定模型参数

例如只用 `best_params.csv` 第 0 行，并只用 fold 0：

```bash
MODEL_ROWS=0 FOLD_MODE=single_fold SINGLE_FOLD=0 \
TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

例如用前 5 行，但只用最新 fold 0：

```bash
MODEL_ROWS=0,1,2,3,4 FOLD_MODE=single_fold SINGLE_FOLD=0 \
TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

默认第一版推荐：

```bash
MODEL_ROWS=0,1,2,3,4 FOLD_MODE=mean_all_folds
```

含义：每个 `model_i` 对 fold0~fold6 的 live 预测取均值，然后五个 `model_i` 再取均值，得到 `ensemble_all5_mean`。

## 输出

```text
14_live_predictions.csv
14_live_predictions_report.json
15_live_rank.csv
15_live_rank_report.json
16_live_trade_signal.csv
16_live_trade_signal_report.json
17_live_signal_pipeline_report.json
```

## 重要说明

历史 `test_preds.h5` 的生成方式是“每个历史日期属于哪个 CV test fold，就用哪个 fold 的 checkpoint 预测那一段”。新交易日没有历史 test fold 归属，因此 v1 默认采用 `mean_all_folds`，即 7 个 fold checkpoint 的均值作为 live 预测。这是不重训前提下最稳妥的兼容版。生产严格版应后续重训 final model 并保存 scaler。
