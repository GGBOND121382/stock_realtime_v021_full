# AS1455 Android Executor MVP (AutoJs6)

目标：让 Android 手机直接从 AS1455 服务器读取已经提交为 READY 的 `r21_best` 交易计划，并在同花顺 Android 客户端中完成**自动填单 + 人工最终确认**。

## 当前边界

这是 **MVP / 真机验证版**，不是无人值守交易程序。

- 服务器不把模型、行情、选股或 sizing 下放到手机；手机只执行 `execution_batch`。
- `dry_run`：自动填写代码、数量和价格，验证实际界面值后打开委托确认框，再自动取消，不会最终提交。
- `live`：自动填写并读回验证，程序可以打开同花顺委托确认框，但**不会点击“确认买入/确认卖出”**；最终确认必须由用户逐笔手动完成。
- 自动填充采用 **初次尝试 + 1 次快速重试**。两次仍失败时，该笔立即跳过并进入人工处理队列，不阻塞后续可安全执行的订单。
- 如果最终确认后的状态无法确定，账本记为 `unknown`。`unknown` 与 `submitted` 一样禁止自动重试，避免重复委托，必须先人工核验。
- 如果同花顺出现无法安全恢复的未知 UI 状态，整批停止，而不是继续盲填。
- BUY 强制 `qty % 100 == 0`；SELL 只要求正整数，以兼容合法零股卖出。

## 快速路径

正常一笔订单的 live 路径为：

1. 进入正确的买入/卖出页。
2. 自动填写证券代码。
3. 确认搜索浮层已经关闭，最终交易页的代码与计划一致。
4. 自动填写数量并从实际 `EditText` 读回校验。
5. 自动填写价格并从实际 `EditText` 读回校验。
6. 自动打开同花顺委托确认框。
7. 用户人工核对并点击最终“确认买入/确认卖出”。
8. 程序识别“委托已提交”，记录合同号（如可识别）、关闭结果框并进入下一笔。

程序不会把 `setText()` 没报错当成“填单成功”；只有界面实际值与计划一致才允许进入下一阶段。

## 失败处理

自动填充失败时：

- 第一次失败：返回交易页并重新定位控件，快速重试 1 次。
- 第二次仍失败：记录 `manual_required`，跳过该笔，继续下一笔。
- 人工确认超时且确认框仍存在：程序取消该确认框，把订单放回人工队列。
- 确认框已经消失但没有识别到明确“委托已提交”：记录 `unknown`，禁止自动重试，避免重复委托。
- 无法关闭卡住的确认/结果弹窗或无法恢复交易页：停止整批，要求人工接管。

## 目录

- `main.js`：入口，拉计划、校验、防重复、逐笔执行、异常队列。
- `lib/plan_client.js`：HTTPS/HTTP 读取只读执行 API。
- `lib/safety.js`：READY、日期、策略、顺序、整手等确定性校验。
- `lib/ledger.js`：本地 `signal_id` 执行账本；`submitted` / `unknown` 为自动化终态。
- `lib/ths_adapter.js`：同花顺 Android UI adapter；负责填充、读回、一次重试和人工确认等待。
- `dump_ths_ui.js`：在目标手机快速检查关键 resource-id 是否存在。
- `config.example.json`：配置模板。

## 关键配置

建议从 `config.example.json` 开始：

```json
{
  "mode": "dry_run",
  "require_manual_confirm": true,
  "ui_timeout_ms": 5000,
  "fill_timeout_ms": 1800,
  "field_verify_timeout_ms": 700,
  "manual_confirm_timeout_ms": 5500,
  "manual_result_grace_ms": 1000,
  "between_orders_ms": 150,
  "failure_skip_ms": 100
}
```

其中：

- `fill_timeout_ms`：单次找控件/代码解析的快速超时。
- `field_verify_timeout_ms`：数量、价格读回校验等待时间。
- `manual_confirm_timeout_ms`：每笔等待用户最终确认的最长时间。
- `manual_result_grace_ms`：确认框消失后等待成功结果框的短暂宽限。
- `between_orders_ms`：正常订单之间的间隔；为两分钟目标默认压缩到 150 ms。
- `failure_skip_ms`：安全跳过失败订单后的短暂停顿。

## 同花顺控件依据

当前主要使用：

- `com.hexin.plat.android:id/auto_stockcode`
- `com.hexin.plat.android:id/dialogplus_view_container`
- `com.hexin.plat.android:id/stockcode_tv`
- `com.hexin.plat.android:id/stockvolume`
- `com.hexin.plat.android:id/stockprice`
- `com.hexin.plat.android:id/btn_transaction`

这些 ID **必须以目标手机当前同花顺版本实际 UI tree 为准核验**。

## 第一次真机测试

1. 安装 AutoJs6 并开启无障碍权限。
2. 安装/登录同花顺，进入交易页面。
3. 把整个 `android_executor_autojs6/` 目录放到手机 AutoJs6 工作目录。
4. 将 `config.example.json` 复制为 `config.json`，保持 `mode=dry_run`。
5. 运行 `dump_ths_ui.js`，确认关键控件存在。
6. 使用测试 batch 运行 `main.js`。
7. 验收每笔代码、数量、价格均正确，确认框能够打开且 dry-run 会取消。
8. 专门构造一次数量/价格填充失败，验证只重试一次，随后跳过并进入人工队列。
9. 专门测试人工确认超时，确认程序会取消确认框并继续。
10. 多轮模拟盘稳定后，再切换 `mode=live`；live 中最终确认仍由用户逐笔点击。

## 服务器 API

`scripts/serve_as1455_execution_api.py` 只读取已经存在的 `execution_batch.json`：

```text
GET /health
GET /api/v1/execution/latest
GET /api/v1/execution/YYYY-MM-DD
```

没有 READY 时返回 `204 No Content`。API 不会生成订单，也不会把非 READY 文件暴露给手机。

## 尚未做

- 没有在你的真实 Android 设备/同花顺版本上完成最终控件验收。
- 没有自动处理登录、验证码、人脸/指纹、安全键盘。
- 没有自动读取券商账户；仍使用服务器侧的实盘账户手动校准。
- 没有无人值守定时唤醒。
- 没有上传成交回报到服务器。

保留这些边界是刻意的：目标是把 14:55 后的人工操作压缩为“核对 + 最终确认”，而不是把最终交易确认交给脚本。
