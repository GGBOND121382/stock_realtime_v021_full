"use strict";

auto.waitFor();

var client = require("./lib/plan_client.js");
var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var retryQueue = require("./lib/retry_queue.js");

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  if (!cfg.api_base_url) throw new Error("config.api_base_url is required");
  if (!cfg.production_experiment) throw new Error("config.production_experiment is required");
  cfg.mode = String(cfg.mode || "dry_run");
  if (cfg.mode !== "live") throw new Error("config.mode must be live for queue_retry_by_code.js");
  return cfg;
}

function normalizeCodes(text) {
  var seen = {};
  var out = [];
  String(text || "").split(/[\s,，;；]+/).forEach(function (raw) {
    var code = String(raw || "").trim();
    if (!code) return;
    if (!/^\d{6}$/.test(code)) throw new Error("invalid stock code: " + code);
    if (!seen[code]) {
      seen[code] = true;
      out.push(code);
    }
  });
  return out;
}

function findOrders(batch, codes) {
  var wanted = {};
  codes.forEach(function (code) { wanted[code] = true; });
  var found = batch.orders.filter(function (order) { return !!wanted[String(order.code)]; });
  var foundCodes = {};
  found.forEach(function (order) { foundCodes[String(order.code)] = true; });
  var missing = codes.filter(function (code) { return !foundCodes[code]; });
  if (missing.length) throw new Error("codes not found in current READY batch: " + missing.join(", "));
  return found;
}

function summary(orders) {
  var lines = [
    "只有在你已经人工核对券商‘当日委托’，确认这些订单实际上没有提交时，才允许解除 ledger 的 submitted 状态。",
    "",
    "将加入重试队列："
  ];
  orders.forEach(function (o) {
    var old = ledger.get(o.signal_id);
    lines.push(
      "#" + o.sequence + " " + o.code + " " + String(o.side).toUpperCase() +
      " x" + o.qty + " @ " + Number(o.submit_price).toFixed(2) +
      "  ledger=" + String(old && old.status || "none")
    );
  });
  lines.push("", "确认后仅这些 signal_id 会被改成 failed，并写入 retry_orders.json；其他已成功订单完全不动。");
  return lines.join("\n");
}

function queueOne(order) {
  var marker = new Error("operator verified order is absent from broker current-day orders");
  marker.stage = "operator_verified_not_submitted";
  marker.ambiguous = false;
  marker.fatal_ui_state = false;

  ledger.markFailed(order, marker.message);
  retryQueue.recordFailure(order, marker, "operator_verified_not_submitted");
}

function run() {
  var config = loadConfig();
  console.show();

  var raw = client.fetchLatest(config);
  if (!raw) throw new Error("current production endpoint returned 204; keep the test batch active before using this helper");
  var batch = safety.validateBatch(raw, config);

  var input = dialogs.input("输入需要强制重试的股票代码", "600641");
  if (input === null || input === undefined) return;
  var codes = normalizeCodes(input);
  if (!codes.length) return;

  var orders = findOrders(batch, codes);
  if (!dialogs.confirm("确认券商确实没有这些委托", summary(orders))) return;

  orders.forEach(queueOne);

  var state = retryQueue.load();
  dialogs.alert(
    "已加入重试队列",
    "已加入: " + orders.length + " 笔\n" +
    orders.map(function (o) {
      return o.code + " x" + o.qty + " @ " + Number(o.submit_price).toFixed(2);
    }).join("\n") +
    "\n\n下一步运行 retry_failed_orders.js\n\n队列文件：\n" + retryQueue.path() +
    "\n当前 JSON 剩余: " + state.orders.length
  );
}

try {
  run();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  dialogs.alert("加入重试队列失败", String(e));
}
