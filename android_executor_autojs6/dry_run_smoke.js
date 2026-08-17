"use strict";

auto.waitFor();

var ths = require("./lib/ths_adapter.js");
var CSV_NAME = "smoke_orders.csv";

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) throw new Error("missing config.json");
  var cfg = JSON.parse(files.read(path));
  return {
    ths_package: String(cfg.ths_package || "com.hexin.plat.android"),
    ui_timeout_ms: Number(cfg.ui_timeout_ms || 5000),
    smoke_csv_row: Number(cfg.smoke_csv_row || 1),
    smoke_mode: String(cfg.smoke_mode || "fill_only")
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
  if (lines.length < 2) throw new Error("smoke CSV has no data rows: " + path);

  var headers = parseCsvLine(lines[0]).map(function (x) { return x.trim(); });
  var required = ["symbol", "side", "shares", "raw_exec_price"];
  required.forEach(function (name) {
    if (headers.indexOf(name) < 0) throw new Error("smoke CSV missing column: " + name);
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

function summary(order, rowNumber, total, mode) {
  var modeText = mode === "live_submit" ? "真实提交一笔" :
    (mode === "confirm_cancel" ? "到确认框后自动取消" : "仅填写，不提交");
  return [
    modeText,
    "CSV行: " + rowNumber + "/" + total,
    "股票: " + order.symbol + " → " + order.code,
    "方向: " + order.side,
    "数量: " + order.qty,
    "价格: " + Number(order.submit_price).toFixed(2)
  ].join("\n");
}

function requireLiveAck(order) {
  var expected = "SUBMIT " + order.code + " " + order.qty + " " + Number(order.submit_price).toFixed(2);
  var entered = dialogs.rawInput(
    "真实委托二次确认\n本模式会真实提交一笔委托。提交后请你立即到同花顺撤单。\n请输入下面整行文本继续：\n" + expected,
    ""
  );
  if (String(entered || "").trim() !== expected) {
    throw new Error("live_submit acknowledgement mismatch; order not submitted");
  }
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
  if (["fill_only", "confirm_cancel", "live_submit"].indexOf(config.smoke_mode) < 0) {
    throw new Error("config.smoke_mode must be fill_only, confirm_cancel, or live_submit");
  }

  var rows = readCsv(csvPath);
  var rowNumber = config.smoke_csv_row;
  if (!isFinite(rowNumber) || Math.floor(rowNumber) !== rowNumber || rowNumber < 1 || rowNumber > rows.length) {
    throw new Error("config.smoke_csv_row must be an integer from 1 to " + rows.length + ": " + rowNumber);
  }

  var order = normalizeOrder(rows[rowNumber - 1], rowNumber);
  var text = summary(order, rowNumber, rows.length, config.smoke_mode);

  if (!dialogs.confirm("AS1455 SMOKE TEST", text)) return;

  var result;
  if (config.smoke_mode === "fill_only") {
    result = ths.fillOrderFields(order, config);
    writeResult("fill_only", rowNumber, order, result);
    toast("FILL-ONLY完成：请人工核对同花顺代码/价格/数量；未点击委托按钮");
    return;
  }

  if (config.smoke_mode === "confirm_cancel") {
    result = ths.preview(order, config);
    writeResult("confirm_cancel", rowNumber, order, result);
    toast("CONFIRM-CANCEL完成：已到确认框并自动取消");
    return;
  }

  requireLiveAck(order);
  result = ths.submit(order, config);
  var resultPath = writeResult("live_submit", rowNumber, order, result);
  dialogs.alert(
    "真实委托已提交",
    "脚本已完成真实委托提交。\n请立即到同花顺“撤单”页面人工撤销该笔测试委托。\n\n结果文件：\n" + resultPath
  );
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 SMOKE TEST 失败", String(e));
}
