# AS1455 Android Executor MVP (AutoJs6)

目标：让 Android 手机直接从 AS1455 服务器读取已经提交为 READY 的 `r21_best` 交易计划，并控制手机上的同花顺 Android 客户端完成下单。

## 当前边界

这是 **MVP / 模拟盘验证版**，不是已经在你的真机同花顺版本上验收过的实盘程序。

- 服务器不把模型、行情、选股或 sizing 下放到手机；手机只执行 `execution_batch`。
- 默认 `mode=dry_run`：会把订单填写到同花顺并走到委托确认框，然后点击取消，不会最终提交。
- `mode=live` 才会点击最终确认；切换前必须先在目标手机/目标同花顺版本验证所有 resource-id。
- 默认要求每次人工确认整个 batch 后才执行。
- 本地用 AutoJs6 `storages` 记录每个 `signal_id`，终态订单不会重复执行。
- BUY 强制 `qty % 100 == 0`；SELL 只要求正整数，以兼容合法零股卖出。

## 目录

- `main.js`：入口，拉计划、校验、防重复、逐笔执行。
- `lib/plan_client.js`：HTTPS/HTTP 读取只读执行 API。
- `lib/safety.js`：READY、日期、策略、顺序、整手等确定性校验。
- `lib/ledger.js`：本地 signal_id 执行账本。
- `lib/ths_adapter.js`：同花顺 Android UI adapter。
- `dump_ths_ui.js`：在目标手机快速检查关键 resource-id 是否存在。
- `config.example.json`：配置模板。

## 同花顺控件依据

MVP 的 resource-id 参考开源 `thsauto` Android 实现，包括：

- `com.hexin.plat.android:id/auto_stockcode`
- `com.hexin.plat.android:id/dialogplus_view_container`
- `com.hexin.plat.android:id/stockcode_tv`
- `com.hexin.plat.android:id/stockvolume`
- `com.hexin.plat.android:id/stockprice`
- `com.hexin.plat.android:id/btn_transaction`
- `com.hexin.plat.android:id/ok_btn`
- `com.hexin.plat.android:id/cancel_btn`
- `com.hexin.plat.android:id/prompt_content`

这些 ID **必须以你的目标手机当前同花顺版本实际 UI tree 为准重新核验**。

## 第一次真机测试

1. 安装 AutoJs6，开启其无障碍权限。
2. 安装/登录同花顺，先进入 `交易 -> 模拟/A股` 页面。
3. 把整个 `android_executor_autojs6/` 目录放到手机 AutoJs6 工作目录。
4. 将 `config.example.json` 复制成 `config.json`，先保持：

```json
"mode": "dry_run",
"require_manual_confirm": true
```

5. 运行 `dump_ths_ui.js`。关键控件在对应买入/卖出页面应能找到。
6. 服务器先用测试 batch；运行 `main.js`。
7. 验收：订单代码、数量、价格都正确，程序到达同花顺委托确认框并自动取消。
8. 至少完成多轮模拟盘测试后，才考虑 `mode=live`。

## 服务器 API

本仓库新增 `scripts/serve_as1455_execution_api.py`，只读取已经存在的 `execution_batch.json`：

```bash
export AS1455_EXECUTION_API_TOKEN='生成一个独立只读长随机token'
.venv_as1455/bin/python scripts/serve_as1455_execution_api.py \
  --host 127.0.0.1 \
  --port 8510
```

接口：

```text
GET /health
GET /api/v1/execution/latest
GET /api/v1/execution/YYYY-MM-DD
```

如果配置 token，客户端需发送：

```text
Authorization: Bearer <token>
```

没有 READY 时返回 `204 No Content`。API 不会生成订单，也不会把非 READY 文件暴露给手机。

生产部署建议由 Nginx 通过 HTTPS 将一个独立路径（例如 `/stock-exec-api/`）反代到 `127.0.0.1:8510`，不要直接把 8510 暴露到公网。

## 尚未做

- 没有在你的真实 Android 设备/同花顺版本上验证控件。
- 没有自动处理登录、验证码、人脸/指纹、安全键盘。
- 没有自动读取券商账户；仍使用服务器“实盘账户手动校准”。
- 没有无人值守定时唤醒；第一版建议 14:55 前打开交易页面后手工运行/确认。
- 没有上传成交回报到服务器。

这些边界是刻意保留的：先验证“手机自己拉 READY -> 正确填写同花顺 -> 正确识别确认/结果”这一条最关键链路。
