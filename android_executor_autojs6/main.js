"use strict";

auto.waitFor();

var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var client = require("./lib/plan_client.js");
var ths = require("./lib/ths_adapter.js");
var mobileGuard = require("./lib/mobile_ui_guard.js");

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json; copy config.example.json first");
  var cfg = JSON.parse(files.read(path));
  if (!cfg.api_base_url) throw new Error("config.api_base_url is required");
  if (!cfg.production_experiment) throw new Error("config.production_experiment is required");
  cfg.mode = String(cfg.mode || "dry_run");
  if (cfg.mode !== "dry_run" && cfg.mode !== "live") throw new Error("mode must be dry_run or live");
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

function runMobileGuard(config, label, includeMetadata) {
  if (config.mobile_preflight_enabled === false) return null;
  var result = mobileGuard.assertReady(config, { include_metadata: includeMetadata === true });
  if (result.warnings.length) {
    console.warn("[MOBILE_GUARD_WARN] " + label + " " + result.warnings.join(" | "));
  }
  console.log(
    "[MOBILE_GUARD_OK] " + label +
    " package=" + result.snapshot.package_name +
    " activity=" + result.snapshot.activity_name +
    " orientation=" + result.snapshot.orientation +
    (result.snapshot.app_version_name ? " version=" + result.snapshot.app_version_name : "")
  );
  return result;
}

function queueItem(order, error) {
  return {
    sequence: order.sequence,
    signal_id: order.signal_id,
    code: order.code,
    side: order.side,
    qty: order.qty,
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
    console.log("[OK] " + order.sequence + " " + order.side + " " + order.code + " x" + order.qty);
    sleep(betweenOrdersMs(config));
    return "completed";
  } catch (e) {
    if (config.mode === "live") {
      ledger.markManualRequired(order, e);
      manualQueue.push(queueItem(order, e));
      console.error(
        "[MANUAL_REQUIRED] " + order.sequence + " " + order.code +
        " stage=" + (e && e.stage ? e.stage : "unknown") +
        " ambiguous=" + !!(e && e.ambiguous === true) +
        " error=" + String(e)
      );
      toast("已跳过 " + order.code + "，加入人工处理队列");

      if (e && e.fatal_ui_state === true) {
        throw e;
      }

      try { ths.recoverToTradingPage(order.side, config); } catch (recoverError) {
        var fatal = new Error(
          "THS failed to recover trading page after skipped order " + order.code +
          ": " + String(recoverError)
        );
        fatal.stage = "post_skip_recovery_failed";
        fatal.ambiguous = false;
        fatal.fatal_ui_state = true;
        throw fatal;
      }
      sleep(failureSkipMs(config));
      return "manual_required";
    }

    ledger.markFailed(order, e);
    manualQueue.push(queueItem(order, e));
    console.error("[DRY_RUN_FAILED] " + order.sequence + " " + order.code + " " + String(e));
    sleep(failureSkipMs(config));
    return "failed";
  }
}

function manualQueueText(queue) {
  if (!queue.length) return "";
  return queue.map(function (item) {
    return "#" + item.sequence + " " + item.code + " x" + item.qty +
      " [" + (item.ambiguous ? "UNKNOWN" : "MANUAL") + "] " + item.stage;
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
    toast("AS1455: 今日订单均已提交或处于 UNKNOWN 人工核验状态");
    console.log("[DONE] all signal_ids are submitted or unknown-terminal");
    return;
  }

  if (config.require_manual_confirm !== false) {
    var ok = dialogs.confirm("AS1455 执行确认", summary(batch, pending));
    if (!ok) {
      console.log("[CANCEL] user cancelled");
      return;
    }
  }

  // Full read-only Android/THS environment check once per batch.
  runMobileGuard(config, "batch_start", true);

  console.log("[RUN] trade_date=" + batch.trade_date + " orders=" + pending.length + " mode=" + config.mode);

  var manualQueue = [];
  var completed = 0;
  var skipped = 0;
  for (var i = 0; i < pending.length; i++) {
    // Lightweight read-only guard before each order. Unknown/stale UI state stops
    // the batch before the next order is touched; it never dismisses a dialog.
    runMobileGuard(config, "before_order_" + pending[i].sequence, false);
    var status = executeOne(pending[i], config, manualQueue);
    if (status === "completed") completed++;
    else if (status === "manual_required" || status === "failed") skipped++;
  }

  var msg = "自动流程完成: " + completed + "/" + pending.length;
  if (manualQueue.length) {
    msg += "\n待人工处理: " + manualQueue.length +
      "\n\n" + manualQueueText(manualQueue);
    dialogs.alert("AS1455 待人工处理", msg);
  } else {
    toast("AS1455: 本次订单处理完成 " + completed + " 笔");
  }

  console.log(
    "[DONE] completed=" + completed +
    " manual_or_failed=" + skipped +
    " queue=" + manualQueue.length
  );
}

try {
  runOnce();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  toast("AS1455执行停止: " + e);
  dialogs.alert("AS1455 执行停止", String(e));
}
