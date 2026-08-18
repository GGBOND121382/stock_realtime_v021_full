"use strict";

auto.waitFor();

var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var client = require("./lib/plan_client.js");
var runner = require("./lib/order_runner.js");
var mobileGuard = require("./lib/mobile_ui_guard.js");
var mobileGuardRunner = require("./lib/mobile_guard_runner.js");
var mobileBaseline = null;

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

function unresolvedLedgerItems(batch) {
  var unresolvedStatuses = {
    unknown: true,
    blocked: true,
    confirmation_pending: true,
    confirmation_open: true
  };
  var items = [];
  batch.orders.forEach(function (order) {
    var item = ledger.get(order.signal_id);
    if (item && unresolvedStatuses[String(item.status || "")]) {
      items.push({ order: order, ledger: item });
    }
  });
  return items;
}

function runMobileGuard(config, label, captureBaseline) {
  if (config.mobile_preflight_enabled === false) return null;
  var settleMs = captureBaseline === true ? Math.min(Number(config.ui_timeout_ms || 1500), 1800) : 500;
  var result = mobileGuardRunner.waitUntilReady(config, settleMs);
  if (captureBaseline === true) {
    mobileBaseline = result.snapshot;
  } else if (mobileBaseline) {
    var stable = mobileGuard.compareBaseline(mobileBaseline, result.snapshot);
    if (!stable.ok) {
      var baselineError = new Error("THS mobile layout changed during batch: " + stable.errors.join("; "));
      baselineError.stage = "mobile_layout_changed";
      baselineError.ambiguous = false;
      baselineError.fatal_ui_state = true;
      throw baselineError;
    }
  }
  if (result.warnings.length) {
    console.warn("[MOBILE_GUARD_WARN] " + label + " " + result.warnings.join(" | "));
  }
  console.log(
    "[MOBILE_GUARD_OK] " + label +
    " package=" + result.snapshot.package_name +
    " activity=" + result.snapshot.activity_name +
    " orientation=" + result.snapshot.orientation +
    " version=" + result.snapshot.app_version_name
  );
  return result;
}

function queueItem(order, kind, detail) {
  var error = detail && detail.error ? detail.error : detail;
  return {
    sequence: order.sequence,
    signal_id: order.signal_id,
    code: order.code,
    side: order.side,
    qty: order.qty,
    kind: kind,
    stage: detail && detail.stage ? String(detail.stage) :
      (error && error.stage ? String(error.stage) : kind),
    ambiguous: !!(error && error.ambiguous === true),
    error: error ? String(error) : ""
  };
}

function ledgerStore() {
  return {
    markStarted: function (order) { ledger.markStarted(order); },
    markConfirmationPending: function (order, detail) { ledger.markConfirmationPending(order, detail); },
    markConfirmationOpen: function (order, detail) { ledger.markConfirmationOpen(order, detail); },
    markResult: function (order, result, dryRun) { ledger.markResult(order, result, dryRun); },
    markRejected: function (order, result) { ledger.markRejected(order, result); },
    markError: function (order, error, dryRun) {
      if (dryRun) ledger.markFailed(order, error);
      else ledger.markManualRequired(order, error);
    }
  };
}

function executeOne(order, config, resultQueue) {
  if (ledger.isTerminal(order.signal_id)) {
    console.log("[SKIP] terminal " + order.signal_id + " " + order.code);
    return "skipped";
  }

  var outcome;
  try {
    outcome = runner.execute(order, config, ledgerStore());
  } catch (e) {
    resultQueue.push(queueItem(order, e && e.ambiguous === true ? "UNKNOWN" : "FATAL", e));
    console.error(
      "[STOP] " + order.sequence + " " + order.code +
      " stage=" + (e && e.stage ? e.stage : "unknown") +
      " ambiguous=" + !!(e && e.ambiguous === true) +
      " error=" + String(e)
    );
    throw e;
  }

  if (outcome.status === "completed") {
    console.log("[OK] " + order.sequence + " " + order.side + " " + order.code + " x" + order.qty);
    sleep(betweenOrdersMs(config));
    return "completed";
  }

  if (outcome.status === "rejected") {
    resultQueue.push(queueItem(order, "REJECTED", {
      stage: outcome.result.stage || "broker_rejected",
      error: outcome.result.prompt && outcome.result.prompt.content ? outcome.result.prompt.content : "broker rejected order"
    }));
    console.error("[REJECTED] " + order.sequence + " " + order.code + " " + JSON.stringify(outcome.result.prompt || {}));
    sleep(failureSkipMs(config));
    return "rejected";
  }

  if (outcome.status === "manual_required" || outcome.status === "failed") {
    resultQueue.push(queueItem(order, "MANUAL", outcome.error));
    console.error(
      "[MANUAL_REQUIRED] " + order.sequence + " " + order.code +
      " stage=" + (outcome.error && outcome.error.stage ? outcome.error.stage : "unknown") +
      " error=" + String(outcome.error)
    );
    sleep(failureSkipMs(config));
    return outcome.status;
  }

  return outcome.status;
}

function resultQueueText(queue) {
  if (!queue.length) return "";
  return queue.map(function (item) {
    return "#" + item.sequence + " " + item.code + " x" + item.qty +
      " [" + item.kind + "] " + item.stage +
      (item.error ? " - " + item.error : "");
  }).join("\n");
}

function requiresManualAttention(item) {
  return item.kind === "MANUAL" || item.kind === "UNKNOWN" || item.kind === "FATAL";
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

  if (config.mode === "live") {
    var unresolved = unresolvedLedgerItems(batch);
    if (unresolved.length) {
      throw new Error(
        "unresolved previous broker-confirmation state: " +
        unresolved.map(function (x) {
          return "#" + x.order.sequence + " " + x.order.code + "(" + x.ledger.status + ")";
        }).join(", ") +
        ". Verify these orders manually before clearing/resolving the local ledger."
      );
    }
  }

  var pending = batch.orders.filter(function (o) { return !ledger.isTerminal(o.signal_id); });
  if (pending.length === 0) {
    toast("AS1455: 今日订单均处于安全终态");
    console.log("[DONE] all signal_ids are safe terminal states");
    return;
  }

  if (config.require_manual_confirm !== false) {
    var ok = dialogs.confirm("AS1455 执行确认", summary(batch, pending));
    if (!ok) {
      console.log("[CANCEL] user cancelled");
      return;
    }
  }

  mobileGuard.waitForTargetPackage(config, Number(config.mobile_return_timeout_ms || 3500), true);
  runMobileGuard(config, "batch_start", true);

  console.log("[RUN] trade_date=" + batch.trade_date + " orders=" + pending.length + " mode=" + config.mode);

  var resultQueue = [];
  var completed = 0;
  var skipped = 0;
  var rejected = 0;
  for (var i = 0; i < pending.length; i++) {
    runMobileGuard(config, "before_order_" + pending[i].sequence, false);
    var status = executeOne(pending[i], config, resultQueue);
    if (status === "completed") completed++;
    else if (status === "rejected") rejected++;
    else if (status === "manual_required" || status === "failed") skipped++;
  }

  var manualCount = resultQueue.filter(requiresManualAttention).length;
  var msg = "自动流程完成: " + completed + "/" + pending.length +
    "\n明确拒单: " + rejected +
    "\n待人工处理: " + manualCount;
  if (resultQueue.length) {
    msg += "\n\n" + resultQueueText(resultQueue);
    dialogs.alert("AS1455 执行结果", msg);
  } else {
    toast("AS1455: 本次订单处理完成 " + completed + " 笔");
  }

  console.log(
    "[DONE] completed=" + completed +
    " rejected=" + rejected +
    " manual_or_failed=" + skipped +
    " manual_attention=" + manualCount +
    " result_items=" + resultQueue.length
  );
}

try {
  runOnce();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  toast("AS1455执行停止: " + e);
  dialogs.alert("AS1455 执行停止", String(e));
}