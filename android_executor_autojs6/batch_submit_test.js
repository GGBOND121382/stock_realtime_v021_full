"use strict";

auto.waitFor();

var runner = require("./lib/order_runner.js");
var mobileGuard = require("./lib/mobile_ui_guard.js");
var mobileGuardRunner = require("./lib/mobile_guard_runner.js");
var batchState = require("./lib/batch_state.js");
var CSV_NAME = "smoke_orders.csv";
var RESULT_NAME = "batch_submit_test_result.json";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  cfg.mode = "live";
  cfg.ths_package = String(cfg.ths_package || "com.hexin.plat.android");
  cfg.ui_timeout_ms = Number(cfg.ui_timeout_ms || 5000);
  cfg.fill_timeout_ms = Number(cfg.fill_timeout_ms || 1800);
  cfg.field_verify_timeout_ms = Number(cfg.field_verify_timeout_ms || 700);
  cfg.manual_confirm_timeout_ms = Number(cfg.manual_confirm_timeout_ms || 5500);
  cfg.manual_result_grace_ms = Number(cfg.manual_result_grace_ms || 1000);
  cfg.between_orders_ms = Number(cfg.between_orders_ms || 150);
  cfg.failure_skip_ms = Number(cfg.failure_skip_ms || 100);
  cfg.mobile_return_timeout_ms = Number(cfg.mobile_return_timeout_ms || 3500);
  cfg.allow_batch_live_test = cfg.allow_batch_live_test === true;
  cfg.batch_test_continue_on_error = cfg.batch_test_continue_on_error !== false;
  return cfg;
}

function sessionDate() {
  return new java.text.SimpleDateFormat("yyyy-MM-dd").format(new java.util.Date());
}

function requireBatchAck(orders, totalNotional, state, stateSource) {
  var buys = orders.filter(function (o) { return o.side === "buy"; }).length;
  var sells = orders.length - buys;
  var completed = state.results.filter(function (x) { return batchState.isSafeTerminal(x.status); }).length;
  var summary = [
    "即将进入 CSV 批量下单测试",
    "订单数: " + orders.length,
    "已安全终态: " + completed,
    "买入: " + buys + " / 卖出: " + sells,
    "计划金额合计: " + Number(totalNotional).toFixed(2) + " 元",
    "状态来源: " + stateSource,
    "",
    "程序逐笔填写并核验确认框；最终确认由你手动点击。",
    "同一交易日、同一 CSV 下，已提交/明确拒单的行不会自动重放。"
  ].join("\n");
  return dialogs.confirm("AS1455 批量下单测试", summary);
}

function makeStore(state, item, resultPath) {
  function persist(status, extra) {
    item.status = status;
    item.updated_at = new Date().toISOString();
    if (extra) {
      Object.keys(extra).forEach(function (key) { item[key] = extra[key]; });
    }
    batchState.persistDurable(resultPath, state);
  }

  return {
    markStarted: function () {
      item.attempts = Number(item.attempts || 0) + 1;
      item.started_at = new Date().toISOString();
      item.error = "";
      item.stage = "started";
      persist("started");
    },
    markConfirmationPending: function (order, detail) {
      persist("confirmation_pending", { stage: "confirmation_phase_started", confirmation: detail || null });
    },
    markConfirmationOpen: function (order, detail) {
      persist("confirmation_open", { stage: "confirmation_open", confirmation: detail || null });
    },
    markResult: function (order, result) {
      persist("submitted", {
        stage: result && result.stage ? result.stage : "submitted",
        broker_result: result || null,
        finished_at: new Date().toISOString()
      });
    },
    markRejected: function (order, result) {
      persist("rejected", {
        stage: result && result.stage ? result.stage : "rejected",
        broker_result: result || null,
        finished_at: new Date().toISOString()
      });
    },
    markError: function (order, error) {
      var ambiguous = !!(error && error.ambiguous === true);
      var fatal = !!(error && error.fatal_ui_state === true);
      persist(ambiguous ? "unknown" : (fatal ? "blocked" : "manual_required"), {
        stage: error && error.stage ? String(error.stage) : "unknown",
        ambiguous: ambiguous,
        fatal_ui_state: fatal,
        error: String(error),
        finished_at: new Date().toISOString()
      });
    }
  };
}

function checkGuard(config, baseline, label) {
  if (config.mobile_preflight_enabled === false) return baseline;
  var settleMs = baseline ? 500 : Math.min(config.ui_timeout_ms, 1800);
  var result = mobileGuardRunner.waitUntilReady(config, settleMs);
  if (baseline) {
    var stable = mobileGuard.compareBaseline(baseline, result.snapshot);
    if (!stable.ok) {
      var err = new Error("THS mobile layout changed during batch: " + stable.errors.join("; "));
      err.stage = "mobile_layout_changed";
      err.ambiguous = false;
      err.fatal_ui_state = true;
      throw err;
    }
  }
  if (result.warnings.length) console.warn("[MOBILE_GUARD_WARN] " + label + " " + result.warnings.join(" | "));
  return baseline || result.snapshot;
}

function run() {
  var config = loadConfig();
  if (!config.allow_batch_live_test) {
    throw new Error("config.allow_batch_live_test must be true for batch real-order testing");
  }

  var csvPath = files.join(files.cwd(), CSV_NAME);
  if (!files.exists(csvPath)) throw new Error("missing " + CSV_NAME);

  var doc = batchState.readCsvText(files.read(csvPath));
  var orders = [];
  var totalNotional = 0;
  for (var i = 0; i < doc.rows.length; i++) {
    var order = batchState.normalizeOrder(doc.rows[i], i + 1);
    orders.push(order);
    totalNotional += order.qty * order.submit_price;
  }
  if (!orders.length) throw new Error("no orders in CSV");
  batchState.validateUniqueOrders(orders);

  var fingerprint = batchState.fingerprintText(doc.text);
  var resultPath = files.join(files.cwd(), RESULT_NAME);
  var loaded = batchState.loadDurable(
    resultPath,
    fingerprint,
    orders,
    totalNotional,
    CSV_NAME,
    sessionDate()
  );
  var state = loaded.state;

  // If the primary file was corrupt but a valid backup was recovered, restore a
  // verified primary before touching broker UI.
  if (loaded.source === "backup") {
    batchState.persistDurable(resultPath, state);
  }

  var unresolved = batchState.unresolvedRows(state);
  if (unresolved.length) {
    throw new Error(
      "previous run has unresolved broker-confirmation state at rows: " +
      unresolved.map(function (x) { return x.row + "(" + x.status + ")"; }).join(", ") +
      ". Verify these orders manually before resetting the batch state."
    );
  }

  if (!requireBatchAck(orders, totalNotional, state, loaded.source)) return;

  mobileGuard.waitForTargetPackage(config, config.mobile_return_timeout_ms, true);
  var guardBaseline = checkGuard(config, null, "batch_start");

  state.status = "running";
  if (!state.started_at) state.started_at = new Date().toISOString();
  batchState.persistDurable(resultPath, state);

  for (var j = 0; j < orders.length; j++) {
    var current = orders[j];
    var item = batchState.itemForRow(state, j + 1, current);
    if (batchState.isSafeTerminal(item.status)) {
      console.log("[SKIP_TERMINAL] row=" + (j + 1) + " status=" + item.status + " code=" + current.code);
      continue;
    }

    checkGuard(config, guardBaseline, "before_order_" + (j + 1));
    toast("处理 " + (j + 1) + "/" + orders.length + " " + current.code);

    var outcome;
    try {
      outcome = runner.execute(current, config, makeStore(state, item, resultPath));
    } catch (e) {
      state.status = "blocked";
      try { batchState.persistDurable(resultPath, state); } catch (persistStopError) {
        console.error("[STATE_PERSIST_FAIL] " + String(persistStopError));
      }
      throw e;
    }

    if (outcome.status === "manual_required") {
      console.error("[MANUAL_REQUIRED] row=" + (j + 1) + " code=" + current.code + " " + String(outcome.error));
      if (!config.batch_test_continue_on_error) {
        state.status = "stopped_on_nonfatal_error";
        batchState.persistDurable(resultPath, state);
        throw outcome.error;
      }
      sleep(config.failure_skip_ms);
      continue;
    }

    if (outcome.status === "rejected") {
      console.error("[REJECTED] row=" + (j + 1) + " code=" + current.code + " " + JSON.stringify(outcome.result.prompt || {}));
      sleep(config.failure_skip_ms);
      continue;
    }

    if (j + 1 < orders.length) sleep(config.between_orders_ms);
  }

  var submitted = state.results.filter(function (x) { return x.status === "submitted"; }).length;
  var rejected = state.results.filter(function (x) { return x.status === "rejected"; }).length;
  var manual = state.results.filter(function (x) { return x.status === "manual_required"; }).length;
  state.status = manual ? "completed_with_manual_items" : "completed";
  state.finished_at = new Date().toISOString();
  batchState.persistDurable(resultPath, state);

  dialogs.alert(
    "批量下单测试完成",
    "已提交: " + submitted + "/" + orders.length +
    "\n明确拒单: " + rejected +
    "\n待人工处理: " + manual +
    "\n\n结果文件：\n" + resultPath
  );
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 批量下单测试失败", String(e));
}