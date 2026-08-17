"use strict";

auto.waitFor();

var safety = require("./lib/safety.js");
var ledger = require("./lib/ledger.js");
var client = require("./lib/plan_client.js");
var ths = require("./lib/ths_adapter.js");

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
    "模式: " + (batch.mode === "live" ? "实盘" : "DRY RUN");
}

function executeOne(order, config) {
  if (ledger.isTerminal(order.signal_id)) {
    console.log("[SKIP] already submitted " + order.signal_id + " " + order.code);
    return;
  }
  ledger.markStarted(order);
  try {
    var result = config.mode === "live" ? ths.submit(order, config) : ths.preview(order, config);
    ledger.markResult(order, result, config.mode !== "live");
    console.log("[OK] " + order.sequence + " " + order.side + " " + order.code + " x" + order.qty);
  } catch (e) {
    ledger.markFailed(order, e);
    throw e;
  }
  sleep(Number(config.between_orders_ms || 800));
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

  // Dry-run records are intentionally repeatable. Only broker-submitted signals
  // are filtered here so a prior dry-run can never suppress a later live order.
  var pending = batch.orders.filter(function (o) { return !ledger.isTerminal(o.signal_id); });
  if (pending.length === 0) {
    toast("AS1455: 今日订单均已实盘提交");
    console.log("[DONE] all signal_ids already submitted");
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
  for (var i = 0; i < pending.length; i++) executeOne(pending[i], config);
  toast("AS1455: 本次订单处理完成 " + pending.length + " 笔");
}

try {
  runOnce();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  toast("AS1455执行失败: " + e);
  dialogs.alert("AS1455 执行失败", String(e));
}
