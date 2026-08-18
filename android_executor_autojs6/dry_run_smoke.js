"use strict";

auto.waitFor();

var ths = require("./lib/ths_adapter.js");
var batch = require("./lib/batch_state.js");
var CSV_NAME = "smoke_orders.csv";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  cfg.ths_package = String(cfg.ths_package || "com.hexin.plat.android");
  cfg.ui_timeout_ms = Number(cfg.ui_timeout_ms || 5000);
  cfg.fill_timeout_ms = Number(cfg.fill_timeout_ms || 1800);
  cfg.field_verify_timeout_ms = Number(cfg.field_verify_timeout_ms || 700);
  cfg.smoke_csv_row = Number(cfg.smoke_csv_row || 1);
  cfg.smoke_mode = String(cfg.smoke_mode || "fill_only");
  return cfg;
}

function summary(order, rowNumber, total, mode) {
  var modeText = mode === "confirm_cancel" ? "填写后打开确认框并自动取消" : "仅填写，不点击委托按钮";
  return [
    modeText,
    "CSV行: " + rowNumber + "/" + total,
    "股票: " + order.symbol + " → " + order.code,
    "方向: " + order.side,
    "数量: " + order.qty,
    "价格: " + Number(order.submit_price).toFixed(2)
  ].join("\n");
}

function writeResult(mode, rowNumber, order, result) {
  var resultPath = files.join(files.cwd(), "smoke_result.txt");
  files.write(resultPath, JSON.stringify({
    status: "ok",
    mode: mode,
    csv_row: rowNumber,
    order: order,
    result: result,
    finished_at: new Date().toISOString()
  }, null, 2));
  return resultPath;
}

function run() {
  var csvPath = files.join(files.cwd(), CSV_NAME);
  if (!files.exists(csvPath)) {
    throw new Error("missing " + CSV_NAME + "; copy the supplied CSV into this folder and rename it to " + CSV_NAME);
  }

  var config = loadConfig();
  if (["fill_only", "confirm_cancel"].indexOf(config.smoke_mode) < 0) {
    throw new Error("config.smoke_mode must be fill_only or confirm_cancel");
  }

  var doc = batch.readCsvText(files.read(csvPath));
  var rowNumber = config.smoke_csv_row;
  if (!isFinite(rowNumber) || Math.floor(rowNumber) !== rowNumber || rowNumber < 1 || rowNumber > doc.rows.length) {
    throw new Error("config.smoke_csv_row must be an integer from 1 to " + doc.rows.length + ": " + rowNumber);
  }

  var order = batch.normalizeOrder(doc.rows[rowNumber - 1], rowNumber);
  if (!dialogs.confirm("AS1455 SMOKE TEST", summary(order, rowNumber, doc.rows.length, config.smoke_mode))) return;

  var result;
  if (config.smoke_mode === "fill_only") {
    result = ths.fillOrderFields(order, config);
    writeResult("fill_only", rowNumber, order, result);
    toast("FILL-ONLY完成：请人工核对代码/数量/价格；脚本没有点击委托按钮");
    return;
  }

  result = ths.preview(order, config);
  writeResult("confirm_cancel", rowNumber, order, result);
  toast("CONFIRM-CANCEL完成：已到确认框并自动取消；未提交委托");
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 SMOKE TEST 失败", String(e));
}
