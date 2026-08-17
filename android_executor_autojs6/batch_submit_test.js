"use strict";

auto.waitFor();

var ths = require("./lib/ths_adapter.js");
var CSV_NAME = "smoke_orders.csv";
var RESULT_NAME = "batch_submit_test_result.json";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  return {
    ths_package: String(cfg.ths_package || "com.hexin.plat.android"),
    ui_timeout_ms: Number(cfg.ui_timeout_ms || 5000),
    between_orders_ms: Number(cfg.between_orders_ms || 800),
    allow_batch_live_test: cfg.allow_batch_live_test === true,
    batch_test_continue_on_error: cfg.batch_test_continue_on_error === true
  };
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
    "即将真实提交 CSV 中全部订单",
    "订单数: " + orders.length,
    "买入: " + buys + " / 卖出: " + sells,
    "计划金额合计: " + notionalText + " 元",
    "",
    "提交后请人工到同花顺撤单页面处理测试委托。"
  ].join("\n");
  if (!dialogs.confirm("AS1455 批量真实下单测试", summary)) return false;

  var expected = "SUBMIT ALL " + orders.length + " ORDERS " + notionalText;
  var entered = dialogs.rawInput(
    "批量真实下单二次确认\n请输入下面整行文本继续：\n" + expected,
    ""
  );
  if (String(entered || "").trim() !== expected) {
    throw new Error("batch live acknowledgement mismatch; no orders submitted");
  }
  return true;
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
    var item = {
      row: j + 1,
      order: current,
      started_at: new Date().toISOString(),
      status: "started"
    };
    state.results.push(item);
    writeState(resultPath, state);

    try {
      toast("提交 " + (j + 1) + "/" + orders.length + " " + current.code);
      var brokerResult = ths.submit(current, config);
      item.status = "submitted";
      item.broker_result = brokerResult;
      item.finished_at = new Date().toISOString();
      writeState(resultPath, state);
    } catch (e) {
      item.status = "failed";
      item.error = String(e);
      item.finished_at = new Date().toISOString();
      state.status = "failed";
      writeState(resultPath, state);
      if (!config.batch_test_continue_on_error) throw e;
    }

    if (j + 1 < orders.length) sleep(config.between_orders_ms);
  }

  var failed = state.results.filter(function (x) { return x.status === "failed"; }).length;
  var submitted = state.results.filter(function (x) { return x.status === "submitted"; }).length;
  state.status = failed ? "completed_with_errors" : "completed";
  state.finished_at = new Date().toISOString();
  writeState(resultPath, state);

  dialogs.alert(
    "批量下单测试完成",
    "已提交: " + submitted + "/" + orders.length +
    "\n失败: " + failed +
    "\n\n请现在到同花顺撤单页面人工处理测试委托。" +
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
