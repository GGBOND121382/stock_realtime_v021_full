"use strict";

var client = require("./lib/plan_client.js");
var safety = require("./lib/safety.js");

var DEFAULT_TEST_EXPERIMENT = "r01_best_reb1_fold0_5_forward";
var RESULT_NAME = "fetch_plan_test_result.json";

function loadTestConfig() {
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
    max_orders: Number(cfg.max_orders || 60)
  };
}

function orderLine(order) {
  return "#" + order.sequence + " " + order.side.toUpperCase() + " " +
    order.code + " x" + order.qty + " @ " + Number(order.submit_price).toFixed(2);
}

function run() {
  var config = loadTestConfig();
  console.show();
  console.log("[FETCH_TEST] api=" + config.api_base_url);
  console.log("[FETCH_TEST] experiment=" + config.production_experiment);

  var raw = client.fetchLatest(config, config.production_experiment);
  if (!raw) {
    var noPlan = "服务器返回 204：今天没有该测试策略的 READY 计划。\n\n" +
      "测试策略: " + config.production_experiment + "\n" +
      "API: " + config.api_base_url;
    console.log("[FETCH_TEST] no READY batch");
    dialogs.alert("AS1455 拉取测试", noPlan);
    return;
  }

  var batch = safety.validateBatch(raw, config);
  var result = {
    status: "ok",
    fetched_at: new Date().toISOString(),
    api_base_url: config.api_base_url,
    experiment: batch.experiment,
    trade_date: batch.trade_date,
    order_count: batch.orders.length,
    orders: batch.orders
  };
  var resultPath = files.join(files.cwd(), RESULT_NAME);
  files.write(resultPath, JSON.stringify(result, null, 2));

  var previewCount = Math.min(batch.orders.length, 10);
  var lines = [
    "手机端拉取 + safety 校验通过",
    "策略: " + batch.experiment,
    "日期: " + batch.trade_date,
    "订单数: " + batch.orders.length,
    "",
    "前 " + previewCount + " 笔:"
  ];
  for (var i = 0; i < previewCount; i++) lines.push(orderLine(batch.orders[i]));
  if (batch.orders.length > previewCount) {
    lines.push("... 其余 " + (batch.orders.length - previewCount) + " 笔省略");
  }
  lines.push("", "结果文件: " + resultPath);

  console.log("[FETCH_TEST] PASS experiment=" + batch.experiment +
    " trade_date=" + batch.trade_date + " orders=" + batch.orders.length);
  dialogs.alert("AS1455 拉取测试通过", lines.join("\n"));
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 拉取测试失败", String(e));
}
