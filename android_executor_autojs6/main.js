"use strict";

auto.waitFor();

var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var client = require("./lib/plan_client.js");
var ths = require("./lib/ths_adapter.js");
var retryQueue = require("./lib/retry_queue.js");

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json; copy config.example.json first");
  var cfg = JSON.parse(files.read(path));
  if (!cfg.api_base_url) throw new Error("config.api_base_url is required");
  if (!cfg.production_experiment) throw new Error("config.production_experiment is required");
  cfg.mode = String(cfg.mode || "dry_run");
  if (cfg.mode !== "dry_run" && cfg.mode !== "live") throw new Error("mode must be dry_run or live");
  cfg.manual_confirm_timeout_ms = Number(cfg.manual_confirm_timeout_ms || 20000);
  return cfg;
}

function summary(batch, pending) {
  var sells = pending.filter(function (o) { return o.side === "sell"; }).length;
  var buys = pending.filter(function (o) { return o.side === "buy"; }).length;
  return "AS1455 " + batch.trade_date + "\n" +
    "策略: " + batch.experiment + "\n" +
    "待执行: " + pending.length + " 笔 (卖 " + sells + " / 买 " + buys + ")\n" +
    "模式: " + (batch.mode === "live" ? "实盘（逐笔人工最终确认）" : "DRY RUN");
}

function betweenOrdersMs(config) {
  if (config.between_orders_ms === undefined || config.between_orders_ms === null) return 150;
  return Math.max(0, Number(config.between_orders_ms));
}

function failureSkipMs(config) {
  if (config.failure_skip_ms === undefined || config.failure_skip_ms === null) return 100;
  return Math.max(0, Number(config.failure_skip_ms));
}

function queueItem(order, error) {
  return {
    sequence: order.sequence,
    signal_id: order.signal_id,
    code: order.code,
    side: order.side,
    qty: order.qty,
    submit_price: order.submit_price,
    stage: error && error.stage ? String(error.stage) : "unknown",
    ambiguous: !!(error && error.ambiguous === true),
    error: String(error)
  };
}

function executeOne(order, config, manualQueue) {
  if (ledger.isTerminal(order.signal_id)) {
    console.log("[SKIP] terminal " + order.signal_id + " " + order.code);
    return "skipped";
  }

  ledger.markStarted(order);

  try {
    var result = config.mode === "live" ? ths.submit(order, config) : ths.preview(order, config);
    ledger.markResult(order, result, config.mode !== "live");
    if (config.mode === "live") retryQueue.remove(order.signal_id);
    console.log("[OK] " + order.sequence + " " + order.side + " " + order.code + " x" + order.qty);
    sleep(betweenOrdersMs(config));
    return "completed";
  } catch (e) {
    if (config.mode === "live") {
      ledger.markManualRequired(order, e);
      retryQueue.recordFailure(order, e, "main");
    } else {
      ledger.markFailed(order, e);
    }

    manualQueue.push(queueItem(order, e));
    console.error(
      "[ORDER_FAILED] " + order.sequence + " " + order.code +
      " stage=" + (e && e.stage ? e.stage : "unknown") +
      " ambiguous=" + !!(e && e.ambiguous === true) +
      " error=" + String(e)
    );

    if (e && (e.ambiguous === true || e.fatal_ui_state === true)) {
      throw e;
    }

    try {
      ths.recoverToTradingPage(order.side, config);
    } catch (recoverError) {
      var fatal = new Error(
        "THS failed to recover after skipped order " + order.code + ": " + String(recoverError)
      );
      fatal.stage = "post_skip_recovery_failed";
      fatal.ambiguous = false;
      fatal.fatal_ui_state = true;
      if (config.mode === "live") retryQueue.recordFailure(order, fatal, "main_recovery");
      throw fatal;
    }

    sleep(failureSkipMs(config));
    return config.mode === "live" ? "manual_required" : "failed";
  }
}

function manualQueueText(queue) {
  if (!queue.length) return "";
  return queue.map(function (item) {
    return "#" + item.sequence + " " + item.code + " x" + item.qty + " @" + Number(item.submit_price).toFixed(2) +
      " [" + (item.ambiguous ? "UNKNOWN" : "RETRY") + "] " + item.stage;
  }).join("\n");
}

function runOnce() {
  var config = loadConfig();
  console.log("[FETCH] " + config.api_base_url);
  var raw = client.fetchLatest(config);
  if (!raw) {
    toast("AS1455: 当前无 READY 计划");
    console.log("[IDLE] no READY batch");
    return;
  }

  var batch = safety.validateBatch(raw, config);
  batch.mode = config.mode;

  if (batch.orders.length === 0) {
    toast("AS1455: r21_best 今日无调仓订单");
    console.log("[IDLE] READY r21_best batch contains zero orders");
    return;
  }

  var pending = batch.orders.filter(function (o) { return !ledger.isTerminal(o.signal_id); });
  if (pending.length === 0) {
    toast("AS1455: 今日订单均已提交或处于 UNKNOWN");
    console.log("[DONE] all signal_ids are terminal");
    return;
  }

  if (config.require_manual_confirm !== false) {
    var ok = dialogs.confirm("AS1455 执行确认", summary(batch, pending));
    if (!ok) {
      console.log("[CANCEL] user cancelled");
      return;
    }
  }

  console.log("[RUN] trade_date=" + batch.trade_date + " orders=" + pending.length + " mode=" + config.mode);

  var manualQueue = [];
  var completed = 0;
  var skipped = 0;

  for (var i = 0; i < pending.length; i++) {
    var status = executeOne(pending[i], config, manualQueue);
    if (status === "completed") completed++;
    else if (status === "manual_required" || status === "failed") skipped++;
  }

  var msg = "自动流程完成: " + completed + "/" + pending.length +
    "\n待重试/人工处理: " + manualQueue.length;
  if (manualQueue.length) {
    msg += "\n\n" + manualQueueText(manualQueue) +
      "\n\n已写入: " + retryQueue.path();
    dialogs.alert("AS1455 执行结果", msg);
  } else {
    toast("AS1455: 本次订单处理完成 " + completed + " 笔");
  }

  console.log(
    "[DONE] completed=" + completed +
    " retry_or_manual=" + skipped +
    " queue=" + manualQueue.length
  );
}

try {
  runOnce();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  toast("AS1455执行停止: " + e);
  dialogs.alert("AS1455 执行停止", String(e) + "\n\n异常订单已保留在 retry_orders.json");
}
