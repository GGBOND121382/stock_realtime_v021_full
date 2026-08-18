"use strict";

auto.waitFor();

var ths = require("./lib/ths_adapter.js");
var mobileGuard = require("./lib/mobile_ui_guard.js");
var CSV_NAME = "smoke_orders.csv";
var RESULT_NAME = "batch_submit_test_result.json";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  cfg.ths_package = String(cfg.ths_package || "com.hexin.plat.android");
  cfg.ui_timeout_ms = Number(cfg.ui_timeout_ms || 5000);
  cfg.fill_timeout_ms = Number(cfg.fill_timeout_ms || 1800);
  cfg.field_verify_timeout_ms = Number(cfg.field_verify_timeout_ms || 700);
  cfg.manual_confirm_timeout_ms = Number(cfg.manual_confirm_timeout_ms || 5500);
  cfg.manual_result_grace_ms = Number(cfg.manual_result_grace_ms || 1000);
  cfg.between_orders_ms = Number(cfg.between_orders_ms || 800);
  cfg.allow_batch_live_test = cfg.allow_batch_live_test === true;
  cfg.batch_test_continue_on_error = cfg.batch_test_continue_on_error === true;
  return cfg;
}

function parseCsvLine(line) {
  var fields = [];
  var current = "";
  var quoted = false;
  for (var i = 0; i < line.length; i++) {
    var ch = line.charAt(i);
    if (ch === '"') {
      if (quoted && i + 1 < line.length && line.charAt(i + 1) === '"') {
        current += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === "," && !quoted) {
      fields.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  fields.push(current);
  return fields;
}

function readCsv(path) {
  var text = files.read(path).replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  var lines = text.split("\n").filter(function (line) { return line.trim().length > 0; });
  if (lines.length < 2) throw new Error("batch CSV has no data rows: " + path);

  var headers = parseCsvLine(lines[0]).map(function (x) { return x.trim(); });
  var required = ["symbol", "side", "shares", "raw_exec_price"];
  required.forEach(function (name) {
    if (headers.indexOf(name) < 0) throw new Error("batch CSV missing column: " + name);
  });

  var rows = [];
  for (var i = 1; i < lines.length; i++) {
    var values = parseCsvLine(lines[i]);
    var row = {};
    for (var j = 0; j < headers.length; j++) row[headers[j]] = values[j] === undefined ? "" : values[j];
    rows.push(row);
  }
  return rows;
}

function normalizeOrder(row, rowNumber) {
  var symbol = String(row.symbol || "").trim();
  var match = symbol.match(/^(\d{6})(?:\.(?:SZ|SH))?$/i);
  if (!match) throw new Error("CSV row " + rowNumber + " invalid symbol: " + symbol);

  var side = String(row.side || "").trim().toLowerCase();
  if (side !== "buy" && side !== "sell") throw new Error("CSV row " + rowNumber + " invalid side: " + side);

  var qty = Number(row.shares);
  if (!isFinite(qty) || qty <= 0 || Math.floor(qty) !== qty) {
    throw new Error("CSV row " + rowNumber + " invalid shares: " + row.shares);
  }
  if (side === "buy" && qty % 100 !== 0) {
    throw new Error("CSV row " + rowNumber + " BUY shares must be multiple of 100: " + qty);
  }

  var price = Number(row.raw_exec_price);
  if (!isFinite(price) || price <= 0) {
    throw new Error("CSV row " + rowNumber + " invalid raw_exec_price: " + row.raw_exec_price);
  }

  return {
    code: match[1],
    symbol: symbol,
    side: side,
    qty: qty,
    submit_price: price,
    sequence: rowNumber
  };
}

function writeState(path, state) {
  state.updated_at = new Date().toISOString();
  files.write(path, JSON.stringify(state, null, 2));
}

function requireBatchAck(orders, totalNotional) {
  var buys = orders.filter(function (o) { return o.side === "buy"; }).length;
  var sells = orders.length - buys;
  var notionalText = Number(totalNotional).toFixed(2);
  var summary = [
    "即将进入 CSV 批量下单测试",
    "订单数: " + orders.length,
    "买入: " + buys + " / 卖出: " + sells,
    "计划金额合计: " + notionalText + " 元",
    "",
    "程序会逐笔自动填写代码、数量、价格并打开确认框；",
    "最终确认仍由你逐笔人工点击。"
  ].join("\n");
  return dialogs.confirm("AS1455 批量下单测试", summary);
}

function checkGuard(config, baseline, label, includeMetadata) {
  if (config.mobile_preflight_enabled === false) return baseline;
  var result = mobileGuard.assertReady(config, { include_metadata: includeMetadata === true });
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

  var rows = readCsv(csvPath);
  var orders = [];
  var totalNotional = 0;
  for (var i = 0; i < rows.length; i++) {
    var order = normalizeOrder(rows[i], i + 1);
    orders.push(order);
    totalNotional += order.qty * order.submit_price;
  }
  if (!orders.length) throw new Error("no orders in CSV");

  if (!requireBatchAck(orders, totalNotional)) return;

  // Read-only Android/THS validation before any order field is touched.
  var guardBaseline = checkGuard(config, null, "batch_start", true);

  var resultPath = files.join(files.cwd(), RESULT_NAME);
  var state = {
    status: "running",
    csv_file: CSV_NAME,
    order_count: orders.length,
    total_notional: totalNotional,
    started_at: new Date().toISOString(),
    results: []
  };
  writeState(resultPath, state);

  for (var j = 0; j < orders.length; j++) {
    var current = orders[j];

    // Ensure package/orientation/display/known blockers did not change between orders.
    checkGuard(config, guardBaseline, "before_order_" + (j + 1), false);

    var item = {
      row: j + 1,
      order: current,
      started_at: new Date().toISOString(),
      status: "started"
    };
    state.results.push(item);
    writeState(resultPath, state);

    try {
      toast("处理 " + (j + 1) + "/" + orders.length + " " + current.code);
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
      state.status = "failed";
      writeState(resultPath, state);

      // Topology/cross-write/unknown UI must stop immediately. Never continue a
      // batch when the mapping or submission state is uncertain.
      if (e && (e.fatal_ui_state === true || e.ambiguous === true)) throw e;
      if (!config.batch_test_continue_on_error) throw e;
    }

    if (j + 1 < orders.length) sleep(config.between_orders_ms);
  }

  var failed = state.results.filter(function (x) { return x.status === "failed" || x.status === "unknown"; }).length;
  var submitted = state.results.filter(function (x) { return x.status === "submitted"; }).length;
  state.status = failed ? "completed_with_errors" : "completed";
  state.finished_at = new Date().toISOString();
  writeState(resultPath, state);

  dialogs.alert(
    "批量下单测试完成",
    "已提交: " + submitted + "/" + orders.length +
    "\n失败/未知: " + failed +
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
