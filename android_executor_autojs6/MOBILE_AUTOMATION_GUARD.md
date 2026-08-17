# Android 同花顺自动化防坑说明

这轮只增加**只读防护**，不扩展真实交易页面的自动填写/点击能力。

## 新增防护

`main.js` 默认在整批开始前做一次完整预检，并在每笔开始前做一次轻量预检。发现异常时会在触碰下一笔之前停止，不会自动关闭未知弹窗或处理验证码。

检查内容：

- 当前 package 必须仍是同花顺；
- 代码、数量、价格、交易按钮对应的关键 Accessibility 节点必须存在；
- 可选锁定同花顺 `version_name`；配置了版本锁但版本号读不到也会失败；
- 可选锁定横竖屏；
- 整批启动后自动冻结当时的分辨率和方向，中途旋转屏幕、进入分屏或显示尺寸变化会熔断；
- 检测验证码、重新登录、登录超时、风险提示、系统维护、网络异常、人脸、指纹、安全键盘；
- 检测上一笔残留的“委托买入/卖出确认”或“委托已提交”结果框；
- `ui_timeout_ms < 1000` 或 `field_verify_timeout_ms < 300` 会给出过于激进的等待参数告警。

## 真机预检

第一次换手机、升级同花顺或调整显示设置后，手动进入正常买入/卖出页，再运行：

```text
mobile_preflight.js
```

它会保存 `ths_mobile_preflight_*.txt`，包含 package、Activity、版本号、分辨率、density DPI、横竖屏、关键节点及 blocker 检查结果。

建议第一次验收通过后，把报告里的版本号写入：

```json
"expected_ths_version_name": "已验收版本号"
```

方向也可以锁定：

```json
"expected_orientation": "portrait"
```

默认配置新增：

```json
"mobile_preflight_enabled": true,
"expected_orientation": "",
"expected_ths_version_name": ""
```

## 测试

纯规则测试：

```bash
node android_executor_autojs6/tests/mobile_ui_guard_test.js
```

覆盖关键节点缺失、blocker、包名错误、版本错误/不可读、方向变化、分辨率变化和等待参数告警。

这些检查不依赖 OCR/截图，也不使用固定坐标。每次同花顺升级或换手机后仍需重新跑 `mobile_preflight.js` 和 `dump_ths_ui.js` 做真机验收。
