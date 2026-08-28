# AS1455 Android Executor MVP (AutoJs6)

目标：让 Android 手机读取已经 READY 的 AS1455 执行计划，在同花顺 Android 客户端中完成**自动填单 + 人工最终确认**。

## 当前边界

这是 MVP / 真机验证版，不是无人值守交易程序。

- 手机只执行已经生成好的订单，不在手机侧计算模型、行情、选股或 sizing。
- `dry_run`：填写代码、数量、价格，验证交易表单和委托确认框后自动取消，不提交。
- `live`：填写并读回验证，打开同花顺委托确认框；**最终“确认买入/确认卖出”由用户逐笔手动点击**。
- 登录、验证码、人脸、指纹、安全键盘、未知弹窗不自动处理。
- 不读取券商账户状态作为高速下单前置条件。

## 单笔执行契约

一笔订单只有满足下面三段一致性才允许进入人工最终确认：

```text
订单计划 == 同花顺交易表单 == 同花顺委托确认框
```

具体流程：

1. 进入正确买入/卖出页。
2. 填写股票代码并确认搜索浮层关闭，最终交易页代码正确。
3. 精确定位数量和价格各自 resource-id 下的 `EditText`。
4. 验证代码/数量/价格三个输入框不是同一节点/同一 bounds。
5. 只通过目标 `UiObject.setText()` 写数量和价格；**禁止全局 `setText()` 焦点回退**。
6. 写数量后复核代码；写价格后再次复核代码、数量和价格。
7. 在点击交易按钮前同步持久化 `confirmation_pending` 防重放状态。
8. 打开委托确认框，解析并再次核对方向、代码、数量、价格。
9. 确认框契约正确后同步持久化 `confirmation_open`。
10. 用户人工点击最终确认。
11. 程序区分 `submitted / rejected / unknown`，关闭已识别结果框并进入下一笔。

确认框字段缺失、代码/数量/价格不一致、输入框串写或未知 UI 都会停止，而不是继续提交。

## 防重复与崩溃恢复

正式 `main.js` 使用同步 ledger。AutoJs6 普通 Storage `put()` 是异步 `apply()`；当前关键状态使用 `putSync()`，确保状态落盘完成后才继续下一次券商侧动作。

以下状态禁止自动重放：

- `submitted`
- `rejected`
- `confirmation_pending`
- `confirmation_open`
- `unknown`
- `blocked`

如果上一轮存在 `confirmation_pending / confirmation_open / unknown / blocked`，**整个 live 批次停止**，必须先人工核对券商状态，不能跳过未决订单继续后面的自动下单。

`batch_submit_test.js` 另有当天 + CSV 指纹作用域的持久化状态文件：

```text
batch_submit_test_result.json
batch_submit_test_result.json.bak
batch_submit_test_result.json.tmp
```

写入采用“临时文件完整写入并回读验证 → 保存上一完整主文件为备份 → 替换主文件 → 再回读验证”。主文件损坏或写入中断时只从有效备份恢复；主备都无法证明状态时直接停止，**不会按第一次运行从头重放**。

旧版 `batch_submit_test_result.json` 若没有 `session_date`，同一 CSV 会 fail closed。确认历史测试委托后再人工清理旧状态文件即可开始新的当天测试。

## 异常分类

- 普通、明确发生在确认阶段之前的填单失败：最多一次快速重试；仍失败后恢复交易页并进入 `manual_required`，后续订单可以继续。
- 明确券商拒单（例如价格/数量非法、资金或股票余额不足等）：记录 `rejected`，不当作 UNKNOWN。
- 确认框消失但无法识别结果、未知结果文案、未知弹窗：`unknown`，整批停止。
- 输入框拓扑错误、字段串写、确认框字段不一致、无法关闭结果框、布局发生变化：`blocked/unknown`，整批停止。
- 人工确认超时但确认框仍开着：**不会自动点取消**，避免和用户刚好点击最终确认形成竞态；改为人工接管并阻止自动重放。

## 手机端 Guard

批次开始和每笔开始前检查：

- 当前 package 为同花顺；
- 关键 Accessibility 节点存在；
- 可选锁定同花顺版本和横竖屏；
- 执行中分辨率、方向、版本不变化；
- 没有验证码、登录异常、风险/网络异常等已知 blocker；
- **不存在任何残留 `dialog_layout`**；
- **不存在股票搜索浮层 `dialogplus_view_container`**。

Guard 会等待页面短暂稳定，而不是某个节点一瞬间没出现就立刻判失败。

## 主要入口

- `main.js`：正式 READY execution batch 入口，带同步 ledger、防重放、统一 runner。
- `batch_submit_test.js`：读取本地 `smoke_orders.csv` 的批量真机测试入口；复用与 `main.js` 相同的单笔 runner。
- `batch_validate.js`：只校验 CSV 和历史批量状态，不访问同花顺 UI、不执行委托。
- `dry_run_smoke.js`：只支持 `fill_only` / `confirm_cancel`；**不再提供 live_submit 旁路**。
- `mobile_preflight.js`：切回同花顺并等待普通交易页稳定后做手机环境预检。
- `dump_ths_ui.js`：导出实际 Accessibility/UI tree 供版本验收。

核心模块：

- `lib/ths_adapter.js`：同花顺字段、确认框和结果框适配。
- `lib/order_contract.js`：确认框三字段契约和结果分类。
- `lib/order_runner.js`：唯一单笔执行状态机。
- `lib/ledger.js`：正式执行同步 ledger。
- `lib/batch_state.js`：CSV 校验和批量测试 crash-safe 状态。
- `lib/mobile_ui_guard.js` / `lib/mobile_guard_runner.js`：只读运行环境 Guard。

## 关键配置

```json
{
  "mode": "dry_run",
  "require_manual_confirm": true,
  "mobile_preflight_enabled": true,
  "mobile_return_timeout_ms": 3500,
  "expected_orientation": "",
  "expected_ths_version_name": "",
  "ui_timeout_ms": 5000,
  "fill_timeout_ms": 1800,
  "field_verify_timeout_ms": 700,
  "manual_confirm_timeout_ms": 5500,
  "manual_result_grace_ms": 1000,
  "between_orders_ms": 150,
  "failure_skip_ms": 100,
  "allow_batch_live_test": false,
  "batch_test_continue_on_error": true
}
```

不要通过把 timeout 压到极低来追求速度；节点条件满足会立即继续，timeout 只是失败上限。

## 第一次真机验收顺序

不要直接上 25 笔。按下面顺序验证：

1. `batch_validate.js`：先验证 CSV、重复订单和历史未决状态。
2. `mobile_preflight.js`：确认当前同花顺版本、方向、分辨率、关键 resource-id。
3. `dry_run_smoke.js` + `smoke_mode=fill_only`：只填一笔，人工观察代码/数量/价格是否完全正确，不点交易按钮。
4. `dry_run_smoke.js` + `smoke_mode=confirm_cancel`：验证委托确认框能解析出正确方向/代码/数量/价格，并自动取消。
5. `batch_submit_test.js`：先用 2–3 笔测试人工最终确认、结果分类、下一笔衔接和状态文件。
6. 验证重启场景：在未提交、确认框打开、明确拒单等不同阶段退出，确认不会误重放。
7. 上述全部通过后再扩到完整 20–25 笔。

## 同花顺控件依据

当前主要使用：

- `com.hexin.plat.android:id/auto_stockcode`
- `com.hexin.plat.android:id/dialogplus_view_container`
- `com.hexin.plat.android:id/stockcode_tv`
- `com.hexin.plat.android:id/stockvolume`
- `com.hexin.plat.android:id/stockprice`
- `com.hexin.plat.android:id/btn_transaction`
- `com.hexin.plat.android:id/dialog_layout`
- `com.hexin.plat.android:id/prompt_content`
- `com.hexin.plat.android:id/ok_btn`
- `com.hexin.plat.android:id/cancel_btn`

这些 ID 必须以目标手机当前同花顺版本实际 UI tree 为准。换手机、升级同花顺或调整显示设置后，应重新运行 `mobile_preflight.js` 和 `dump_ths_ui.js`。

## 尚未完成的验证

- 当前代码已经做了静态/纯逻辑测试，但**还没有在你的真实 Android 手机和当前同花顺版本上完成最终端到端验收**。
- 真机 UI tree、券商定制界面、结果文案和弹窗形态仍需按上面的分阶段测试验证。
- 没有自动处理验证码/生物认证/安全键盘，也没有无人值守定时唤醒。
