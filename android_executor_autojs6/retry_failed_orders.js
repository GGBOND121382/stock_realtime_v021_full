"use strict";

auto.waitFor();

var client = require("./lib/plan_client.js");
var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var ths = require("./lib/ths_adapter.js");
var retryQueue = require("./lib/retry_queue.js");

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  if (!cfg.api_base_url) throw new Error("config.api_base_url is required");
  if (!cfg.production_experiment) throw new Error("config.production_experiment is required");
  cfg.mode = String(cfg.mode || "dry_run");
  if (cfg.mode !== "live") throw new Error("config.mode must be live for retry_failed_orders.js");
  cfg.manual_confirm_timeout_ms = Number(cfg.manual_confirm_timeout_ms || 20000);
  cfg.manual_takeover_timeout_ms = Number(cfg.manual_takeover_timeout_ms || 60000);
  cfg.between_orders_ms = Number(cfg.between_orders_ms || 150);
  cfg.failure_skip_ms = Number(cfg.failure_skip_ms || 100);
  return cfg;
}

function bootstrapFromCurrentBatch(config) {
  var state = retryQueue.load();
  if (state.orders.length) return 0;

  var raw = client.fetchLatest(config);
  if (!raw) return 0;
  var batch = safety.validateBatch(raw, config);
  var added = 0;

  for (var i = 0; i < batch.orders.length; i++) {
    var order = batch.orders[i];
    var item = ledger.get(order.signal_id);
    if (!item) continue;
    var status = String(item.status || "");
    if (status === "manual_required" || status === "failed" || status === "unknown" || status === "started") {
      var before = retryQueue.load().orders.length;
      retryQueue.recordFromLedger(order, item);
      if (retryQueue.load().orders.length > before) added++;
    }
  }
  return added;
}

function pruneSubmitted() {
  var state = retryQueue.load();
  var removed = 0;
  var snapshot = state.orders.slice();
  for (var i = 0; i < snapshot.length; i++) {
    var item = ledger.get(snapshot[i].signal_id);
    if (item && String(item.status) === "submitted") {
      retryQueue.remove(snapshot[i].signal_id);
      removed++;
    }
  }
  return removed;
}

function ledgerBlocksAutomaticRetry(item) {
  var stored = ledger.get(item.signal_id);
  if (!stored) return false;
  var status = String(stored.status || "");
  return status === "submitted" || status === "unknown" || status === "started";
}

function retryableItems() {
  return retryQueue.load().orders.filter(function (item) {
    return item.retryable === true && !item.ambiguous && !ledgerBlocksAutomaticRetry(item);
  });
}

function queueSummary(items) {
  var lines = [
    "本次仅重试 retry_orders.json 中明确失败的订单。",
    "每笔只自动尝试一次；成功后立即从 JSON 删除。",
    "UNKNOWN / 状态不确定的订单不会自动重试。",
    "",
    "待重试: " + items.length + " 笔"
  ];
  for (var i = 0; i < items.length; i++) {
    var o = items[i];
    lines.push(
      "#" + o.sequence + " " + o.code + " " + String(o.side).toUpperCase() +
      " x" + o.qty + " @ " + Number(o.submit_price).toFixed(2)
    );
  }
  return lines.join("\n");
}

function tryRecover(order, config) {
  try {
    ths.recoverToTradingPage(order.side, config);
    sleep(config.failure_skip_ms);
    return true;
  } catch (recoverError) {
    var fatal = new Error("THS retry recovery failed for " + order.code + ": " + String(recoverError));
    fatal.stage = "retry_recovery_failed";
    fatal.ambiguous = false;
    fatal.fatal_ui_state = true;
    retryQueue.recordFailure(order, fatal, "retry_recovery");
    throw fatal;
  }
}

function run() {
  var config = loadConfig();
  console.show();

  var bootstrapped = bootstrapFromCurrentBatch(config);
  var pruned = pruneSubmitted();
  var state = retryQueue.load();

  console.log("[RETRY] queue=" + retryQueue.path() +
    " bootstrapped=" + bootstrapped + " pruned_submitted=" + pruned +
    " remaining=" + state.orders.length);

  if (!state.orders.length) {
    dialogs.alert("AS1455 重试", "当前没有异常订单。\n\n" + retryQueue.path());
    return;
  }

  var items = retryableItems();
  var blocked = state.orders.length - items.length;
  if (!items.length) {
    dialogs.alert(
      "AS1455 重试",
      "异常队列中没有可安全自动重试的订单。\n" +
      "UNKNOWN/状态不确定: " + blocked + "\n\n请先人工核对当日委托。\n" + retryQueue.path()
    );
    return;
  }

  var text = queueSummary(items);
  if (blocked > 0) text += "\n\n另有 " + blocked + " 笔 UNKNOWN/不可自动重试，已保留。";
  if (!dialogs.confirm("AS1455 异常订单重试", text)) return;

  var submitted = 0;
  var failed = 0;

  for (var i = 0; i < items.length; i++) {
    var order = items[i];
    var currentLedger = ledger.get(order.signal_id);
    if (currentLedger && String(currentLedger.status) === "submitted") {
      retryQueue.remove(order.signal_id);
      continue;
    }
    if (currentLedger && (String(currentLedger.status) === "unknown" || String(currentLedger.status) === "started")) {
      continue;
    }

    ledger.markStarted(order);
    try {
      toast("重试 " + (i + 1) + "/" + items.length + " " + order.code);
      var result = ths.submit(order, config);
      ledger.markResult(order, result, false);
      retryQueue.remove(order.signal_id);
      submitted++;
      console.log("[RETRY_OK] " + order.code + " x" + order.qty + " @" + Number(order.submit_price).toFixed(2));
    } catch (e) {
      ledger.markManualRequired(order, e);
      retryQueue.recordFailure(order, e, "retry");
      failed++;
      console.error(
        "[RETRY_FAILED] " + order.code +
        " stage=" + (e && e.stage ? e.stage : "unknown") +
        " ambiguous=" + !!(e && e.ambiguous === true) +
        " error=" + String(e)
      );

      if (e && (e.ambiguous === true || e.fatal_ui_state === true)) {
        dialogs.alert(
          "AS1455 重试停止",
          "当前订单状态不确定，已停止后续自动重试，避免重复委托。\n\n" +
          order.code + " x" + order.qty + " @ " + Number(order.submit_price).toFixed(2) +
          "\n\n剩余订单保留在：\n" + retryQueue.path()
        );
        return;
      }
      tryRecover(order, config);
    }

    if (i + 1 < items.length) sleep(config.between_orders_ms);
  }

  var remaining = retryQueue.load().orders;
  dialogs.alert(
    "AS1455 重试完成",
    "本轮成功: " + submitted +
    "\n本轮仍失败: " + failed +
    "\nJSON 剩余: " + remaining.length +
    "\n\n成功订单已从 JSON 删除。\n仍失败的可再次运行本脚本，或手动下单。\n\n" + retryQueue.path()
  );
}

try {
  run();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  dialogs.alert("AS1455 重试失败", String(e) + "\n\n队列未清空：\n" + retryQueue.path());
}
