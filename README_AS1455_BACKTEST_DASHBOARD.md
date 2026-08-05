# AS1455 九模型回测前端

该前端读取：

```text
saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1
```

展示 `r01/r05/r21 × all5/first3/best` 共9组回测的统一指标、Forward收益曲线、历史Fold收益、历史Grid Top20、调仓日期和调仓完成后的持仓。

## 安装

```bash
cd /root/stock_realtime_v021_full
.venv_as1455/bin/pip install -r requirements-dashboard.txt
```

## 启动

```bash
HOST=0.0.0.0 PORT=8501 \
bash scripts/run_as1455_backtest_dashboard.sh
```

可选：为两个刷新按钮增加口令保护：

```bash
AS1455_DASHBOARD_REFRESH_TOKEN='替换为强口令' \
HOST=0.0.0.0 PORT=8501 \
bash scripts/run_as1455_backtest_dashboard.sh
```

浏览器访问：

```text
http://服务器IP:8501
```

建议只允许可信来源访问8501端口，或通过SSH端口转发：

```bash
ssh -L 8501:127.0.0.1:8501 root@服务器
```

服务器端对应启动：

```bash
HOST=127.0.0.1 PORT=8501 bash scripts/run_as1455_backtest_dashboard.sh
```

## 前端刷新

页面提供两种刷新：

1. 更新每日行情与 `forward model_data`，再刷新9组回测；
2. 复用当前 `forward model_data`，仅刷新9组回测。

后台任务由：

```bash
bash scripts/run_as1455_dashboard_refresh.sh
```

执行，状态和日志保存在：

```text
refresh_all_v1/.dashboard/
├── refresh_status.json
├── refresh.lock
└── refresh_YYYYmmdd_HHMMSS.log
```

包装器使用文件锁防止并发，并在任何行情更新之前逐项检查9组历史Grid。任一历史结果缺失时立即失败，不会静默重跑30/150/630组历史Grid。

## 工作日自动刷新

默认安装为北京时间每周一至周五18:30执行：

```bash
bash scripts/install_as1455_dashboard_daily_refresh_cron.sh
```

自定义时间，例如每个工作日20:00：

```bash
CRON_SCHEDULE='0 20 * * 1-5' \
bash scripts/install_as1455_dashboard_daily_refresh_cron.sh
```

安装器只写入独立文件：

```text
/etc/cron.d/as1455-dashboard-refresh
```

不会修改用户已有的 crontab。重复执行会覆盖该独立任务文件。

## 命令行刷新

```bash
# 包含每日数据更新
MATRIX_ROOT=saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1 \
SKIP_DATA_REFRESH=0 \
bash scripts/run_as1455_dashboard_refresh.sh

# 复用当前forward model_data
MATRIX_ROOT=saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1 \
SKIP_DATA_REFRESH=1 \
bash scripts/run_as1455_dashboard_refresh.sh
```
