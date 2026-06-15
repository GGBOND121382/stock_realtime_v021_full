# A-share Chapter 17 NN Reproduction

Runs the Chapter 17 neural-network training flow on the A-share Chapter 12 style model data.

Input:

```text
saved_data/ashare_ml4t/ch12_reproduce/model_data.h5::/model_data
```

Outputs:

```text
saved_data/ashare_ml4t/ch17_reproduce/results/scores.h5::/ic_by_day
saved_data/ashare_ml4t/ch17_reproduce/results/test_preds.h5::/predictions
saved_data/ashare_ml4t/ch17_reproduce/results/logs/
```

This script only changes paths and `--train-end`. It does not change labels, add A-share trading masks, use board fields, apply limit-up/down logic, or switch to long-only trading.

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Smoke test:

```bash
python3 scripts/run_ashare_ch17_nn_reproduce.py --smoke
```

Full run:

```bash
python3 scripts/run_ashare_ch17_nn_reproduce.py
```

With explicit train end:

```bash
python3 scripts/run_ashare_ch17_nn_reproduce.py --train-end 2025
```

In `tmux` with logs:

```bash
mkdir -p logs
python3 scripts/run_ashare_ch17_nn_reproduce.py \
  2>&1 | tee logs/run_ashare_ch17_nn_reproduce_$(date +%Y%m%d_%H%M%S).log
```

Retrain even when `scores.h5` exists:

```bash
python3 scripts/run_ashare_ch17_nn_reproduce.py --force-train
```

Default behavior matches the uploaded Chapter 17 code: if `results/scores.h5` exists, CV training is skipped and existing scores/checkpoints are used to select top models and generate predictions.

## Fixed Training Contract

Data read:

```python
data = pd.read_hdf(model_data_path, "model_data").dropna().sort_index()
outcomes = data.filter(like="fwd").columns.tolist()
X_cv = data.loc[idx[:, :TRAIN_END], :].drop(outcomes, axis=1)
y_cv = data.loc[idx[:, :TRAIN_END], "r01_fwd"]
```

CV:

```text
lookahead = 1
n_splits = 12
train_period_length = 21 * 12 * 4
test_period_length = 21 * 3
```

Parameter grid:

```text
dense_layers = (16, 8), (32, 16), (32, 32), (64, 32)
activation = tanh
dropout = 0, 0.1, 0.2
batch_size = 64, 256
epochs = 20
```

Reports:

```text
train_data_summary.json
cv_split_report.csv
param_grid.csv
scores_summary.csv
best_params.csv
predictions_summary.json
```
