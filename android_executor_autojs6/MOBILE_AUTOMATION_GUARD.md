# Android 同花顺手机端 Guard

本 Guard 的职责是**在触碰下一笔订单之前证明手机 UI 仍处于可预期状态**。它不处理验证码、登录、人脸/指纹、安全键盘，也不会自动关闭未知弹窗。

## 每批 / 每笔检查

- 当前前台 package 必须是同花顺；AutoJs6 自己的确认对话框关闭后会先等待同花顺重新成为前台，必要时再启动同花顺。
- `auto_stockcode`、`stockvolume`、`stockprice`、`btn_transaction` 等关键 Accessibility 节点必须可观察。
- 可选锁定同花顺 `version_name`；版本号读不到或版本变化都失败。
- 可选锁定横竖屏。
- 批次开始后冻结版本、方向和显示尺寸；中途旋转屏幕、进入分屏或显示尺寸变化会停止。
- 已知验证码、重新登录、登录超时、风险提示、系统维护、网络异常、人脸、指纹、安全键盘会停止。
- **任何残留 `dialog_layout` 都会停止**，不依赖弹窗文案是否已收录。
- **股票搜索浮层 `dialogplus_view_container` 残留也会停止**。
- 页面刚切换时允许短暂等待关键节点出现，不用极短固定 sleep 判断成功/失败。

## 和订单状态机的关系

Guard 只负责“下一笔能不能开始”。真正的单笔安全契约在 `ths_adapter.js + order_runner.js`：

```text
订单计划 == 交易表单 == 委托确认框
```

确认框会重新解析并核对方向、代码、数量和价格；不一致时不会进入人工最终确认。

正式 live 模式还会在点击交易按钮前同步落盘 `confirmation_pending`，确认框契约通过后同步落盘 `confirmation_open`。这两个状态以及 `unknown/blocked` 都禁止自动重放。

## 真机预检

第一次换手机、升级同花顺或调整显示设置后运行：

```text
mobile_preflight.js
```

脚本会先切回同花顺并等待普通买入/卖出页稳定，再保存：

```text
ths_mobile_preflight_*.txt
```

内容包括 package、Activity、版本、分辨率、density DPI、横竖屏、关键节点和 blocker。

验收后建议配置：

```json
"mobile_preflight_enabled": true,
"mobile_return_timeout_ms": 3500,
"expected_ths_version_name": "已验收版本号",
"expected_orientation": "portrait"
```

## 测试

纯规则测试：

```text
node android_executor_autojs6/tests/mobile_ui_guard_test.js
```

还应配合：

```text
node android_executor_autojs6/tests/order_contract_test.js
node android_executor_autojs6/tests/batch_state_test.js
node android_executor_autojs6/tests/order_runner_test.js
```

真机仍必须跑：

1. `mobile_preflight.js`
2. `dry_run_smoke.js` / `fill_only`
3. `dry_run_smoke.js` / `confirm_cancel`
4. 2–3 笔 `batch_submit_test.js`
5. 再扩到完整批次

这些检查不依赖 OCR/截图，也不使用固定坐标作为输入字段主定位。每次同花顺升级或换手机后都应重新验收 UI tree。
