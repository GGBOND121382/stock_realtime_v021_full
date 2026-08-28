"use strict";

auto.waitFor();

var client = require("./lib/plan_client.js");
var safety = require("./lib/safety.js");
var ths = require("./lib/ths_adapter.js");

var DEFAULT_TEST_EXPERIMENT = "r01_best_reb1_fold0_5_forward";
var RESULT_NAME = "fetch_submit_test_result.json";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));

  var baseUrl = String(cfg.api_base_url || "").trim();
  if (!baseUrl) throw new Error("config.api_base_url is required");

  var experiment = String(cfg.fetch_test_experiment || DEFAULT_TEST_EXPERIMENT).trim();
  if (!experiment) throw new Error("config.fetch_test_experiment is required");

  return {
    api_base_url: baseUrl,
    api_token: String(cfg.api_token || ""),
    request_timeout_ms: Number(cfg.request_timeout_ms || 5000),
    production_experiment: experiment,
    max_orders: Number(cfg.max_orders || 60),
    ths_package: String(cfg.ths_package || "com.hexin.plat.android"),
    ui_timeout_ms: Number(cfg.ui_timeout_ms || 5000),
    fill_timeout_ms: Number(cfg.fill_timeout_ms || 1800),
    field_verify_timeout_ms: Number(cfg.field_verify_timeout_ms || 700),
    manual_confirm_timeout_ms: Number(cfg.manual_confirm_timeout_ms || 20000),
    manual_result_grace_ms: Number(cfg.manual_result_grace_ms || 1000),
    manual_takeover_timeout_ms: Number(cfg.manual_takeover_timeout_ms || 60000),
    between_orders_ms: Number(cfg.between_orders_ms || 150),
    failure_skip_ms: Number(cfg.failure_skip_ms || 100),
    allow_batch_live_test: cfg.allow_batch_live_test === true,
    batch_test_continue_on_error: cfg.batch_test_continue_on_error !== false
  };
}

function writeState(path, state) {
  state.updated_at = new Date().toISOString();
  files.write(path, JSON.stringify(state, null, 2));
}

function requireBatchAck(batch) {
  var buys = batch.orders.filter(function (o) { return o.side === "buy"; }).length;
  var sells = batch.orders.length - buys;
  var totalNotional = 0;
  batch.orders.forEach(function (o) {
    totalNotional += Number(o.qty) * Number(o.submit_price);
  });

  var summary = [
    "即将执行服务器策略批量下单测试",
    "策略: " + batch.experiment,
    "日期: " + batch.trade_date,
    "订单数: " + batch.orders.length,
    "买入: " + buys + " / 卖出: " + sells,
    "提交价名义金额: " + totalNotional.toFixed(2) + " 元",
    "",
    "程序逐笔填写并打开同花顺确认框；最终确认由你手动点击。",
    "发现自动填充有误时，可点取消并手动修改当前单；脚本等待提交结果后继续。",
    "单笔普通失败会跳过，不会重新填写同一笔。",
    "UNKNOWN/界面状态不确定时会立即停止，避免重复提交。"
  ].join("\n");
  return dialogs.confirm("AS1455 服务器策略下单测试", summary);
}

function run() {
  var config = loadConfig();
  if (!config.allow_batch_live_test) {
    throw new Error("config.allow_batch_live_test must be true for server strategy real-order testing");
  }

  console.show();
  console.log("[FETCH_SUBMIT_TEST] api=" + config.api_base_url);
  console.log("[FETCH_SUBMIT_TEST] experiment=" + config.production_experiment);

  var raw = client.fetchLatest(config, config.production_experiment);
  if (!raw) {
    dialogs.alert(
      "AS1455 服务器策略下单测试",
      "服务器返回 204：今天没有该测试策略的 READY 计划。\n\n策略: " +
        config.production_experiment
    );
    return;
  }

  // Ongoing r01 simulation must consume only the post-14:55 committed batch.
  // The API's historical read-only fallback is useful for diagnostics, but it
  // must never be treated as today's executable simulation signal.
  if (raw.temporary_test_batch === true) {
    dialogs.alert(
      "AS1455 服务器策略下单测试",
      "服务器尚未生成今天正式提交的模拟盘 READY。\n\n" +
        "当前响应只是历史/临时计划回退，已拒绝执行。请稍后重新运行。\n\n策略: " +
        config.production_experiment
    );
    console.log("[IDLE] rejected temporary_test_batch; waiting for committed READY");
    return;
  }

  var batch = safety.validateBatch(raw, config);
  if (!batch.orders.length) {
    dialogs.alert(
      "AS1455 服务器策略下单测试",
      "策略已成功拉取并通过 safety 校验，但今天没有订单。\n\n策略: " + batch.experiment
    );
    return;
  }

  if (!requireBatchAck(batch)) {
    console.log("[CANCEL] user cancelled before first order");
    return;
  }

  var resultPath = files.join(files.cwd(), RESULT_NAME);
  var state = {
    status: "running",
    source: "execution_api_experiment_query_committed_ready",
    api_base_url: config.api_base_url,
    experiment: batch.experiment,
    trade_date: batch.trade_date,
    order_count: batch.orders.length,
    started_at: new Date().toISOString(),
    results: []
  };
  writeState(resultPath, state);

  for (var i = 0; i < batch.orders.length; i++) {
    var current = batch.orders[i];
    var item = {
      sequence: current.sequence,
      signal_id: current.signal_id,
      code: current.code,
      side: current.side,
      qty: current.qty,
      submit_price: current.submit_price,
      started_at: new Date().toISOString(),
      status: "started"
    };
    state.results.push(item);
    writeState(resultPath, state);

    try {
      toast("处理 " + (i + 1) + "/" + batch.orders.length + " " + current.code);
      var brokerResult = ths.submit(current, config);
      item.status = "submitted";
      item.broker_result = brokerResult;
      item.finished_at = new Date().toISOString();
      writeState(resultPath, state);
    } catch (e) {
      item.status = e && e.ambiguous === true ? "unknown" : "failed";
      item.stage = e && e.stage ? String(e.stage) : "unknown";
      item.error = String(e);
      item.finished_at = new Date().toISOString();
      writeState(resultPath, state);

      if (e && (e.ambiguous === true || e.fatal_ui_state === true)) {
        state.status = "blocked";
        writeState(resultPath, state);
        throw e;
      }

      if (!config.batch_test_continue_on_error) {
        state.status = "stopped_on_error";
        writeState(resultPath, state);
        throw e;
      }

      try {
        ths.recoverToTradingPage(current.side, config);
      } catch (recoverError) {
        state.status = "blocked";
        writeState(resultPath, state);
        var fatal = new Error(
          "THS failed to recover after order " + current.sequence + " " +
          current.code + ": " + String(recoverError)
        );
        fatal.stage = "post_skip_recovery_failed";
        fatal.fatal_ui_state = true;
        throw fatal;
      }
      sleep(config.failure_skip_ms);
    }

    if (i + 1 < batch.orders.length) sleep(config.between_orders_ms);
  }

  var submitted = state.results.filter(function (x) { return x.status === "submitted"; }).length;
  var failed = state.results.filter(function (x) { return x.status === "failed"; }).length;
  var unknown = state.results.filter(function (x) { return x.status === "unknown"; }).length;
  state.status = failed ? "completed_with_errors" : "completed";
  state.finished_at = new Date().toISOString();
  writeState(resultPath, state);

  dialogs.alert(
    "服务器策略下单测试完成",
    "策略: " + batch.experiment +
    "\n已提交: " + submitted + "/" + batch.orders.length +
    "\n失败并跳过: " + failed +
    "\nUNKNOWN: " + unknown +
    "\n\n结果文件：\n" + resultPath
  );
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 服务器策略下单测试失败", String(e));
}
