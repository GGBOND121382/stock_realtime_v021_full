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
    smoke_csv_row: Number(cfg.smoke_csv_row || 1)
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

function run() {
  var csvPath = files.join(files.cwd(), CSV_NAME);
  if (!files.exists(csvPath)) {
    throw new Error("missing " + CSV_NAME + "; copy the supplied CSV into this folder and rename it to " + CSV_NAME);
  }

  var config = loadConfig();
  var rows = readCsv(csvPath);
  var rowNumber = config.smoke_csv_row;
  if (!isFinite(rowNumber) || Math.floor(rowNumber) !== rowNumber || rowNumber < 1 || rowNumber > rows.length) {
    throw new Error("config.smoke_csv_row must be an integer from 1 to " + rows.length + ": " + rowNumber);
  }

  var order = normalizeOrder(rows[rowNumber - 1], rowNumber);
  var summary = [
    "仅填写，不提交",
    "CSV行: " + rowNumber + "/" + rows.length,
    "股票: " + order.symbol + " → " + order.code,
    "方向: " + order.side,
    "数量: " + order.qty,
    "价格: " + Number(order.submit_price).toFixed(2),
    "",
    "本测试不会点击同花顺的买入/卖出委托按钮。"
  ].join("\n");

  if (!dialogs.confirm("AS1455 FILL-ONLY 测试", summary)) return;

  var result = ths.fillOrderFields(order, config);
  var resultPath = files.join(files.cwd(), "smoke_fill_result.txt");
  files.write(resultPath, JSON.stringify({
    status: "ok",
    mode: "fill_only",
    csv_row: rowNumber,
    order: order,
    result: result,
    finished_at: new Date().toISOString()
  }, null, 2));

  toast("FILL-ONLY完成：请人工核对同花顺代码/价格/数量；未点击委托按钮");
}

try {
  run();
} catch (e) {
  var message = e && e.stack ? e.stack : String(e);
  console.show();
  console.error(message);
  dialogs.alert("AS1455 FILL-ONLY 测试失败", String(e));
}
