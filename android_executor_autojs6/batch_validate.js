"use strict";

var batch = require("./lib/batch_state.js");

var CSV_NAME = "smoke_orders.csv";
var RESULT_NAME = "batch_submit_test_result.json";
var REPORT_NAME = "batch_validate_result.json";

function sessionDate() {
  return new java.text.SimpleDateFormat("yyyy-MM-dd").format(new java.util.Date());
}

function run() {
  var csvPath = files.join(files.cwd(), CSV_NAME);
  if (!files.exists(csvPath)) throw new Error("missing " + CSV_NAME);

  var doc = batch.readCsvText(files.read(csvPath));
  var orders = [];
  var totalNotional = 0;
  for (var i = 0; i < doc.rows.length; i++) {
    var order = batch.normalizeOrder(doc.rows[i], i + 1);
    orders.push(order);
    totalNotional += order.qty * order.submit_price;
  }
  if (!orders.length) throw new Error("no orders in CSV");
  batch.validateUniqueOrders(orders);

  var fingerprint = batch.fingerprintText(doc.text);
  var previousPath = files.join(files.cwd(), RESULT_NAME);
  var loaded = batch.loadDurable(
    previousPath,
    fingerprint,
    orders,
    totalNotional,
    CSV_NAME,
    sessionDate()
  );
  var state = loaded.state;
  var unresolved = batch.unresolvedRows(state);

  var report = {
    status: unresolved.length ? "BLOCKED_UNRESOLVED" : "OK",
    csv_file: CSV_NAME,
    csv_fingerprint: fingerprint,
    session_date: state.session_date,
    order_count: orders.length,
    buy_count: orders.filter(function (x) { return x.side === "buy"; }).length,
    sell_count: orders.filter(function (x) { return x.side === "sell"; }).length,
    total_notional: totalNotional,
    state_source: loaded.source,
    previous_safe_terminal_rows: state.results.filter(function (x) {
      return batch.isSafeTerminal(x.status);
    }).map(function (x) {
      return { row: x.row, status: x.status, stage: x.stage || "" };
    }),
    previous_unresolved_rows: unresolved.map(function (item) {
      return { row: item.row, status: item.status, stage: item.stage || "" };
    }),
    orders: orders,
    validated_at: new Date().toISOString()
  };

  var reportPath = files.join(files.cwd(), REPORT_NAME);
  files.write(reportPath, JSON.stringify(report, null, 2));

  if (unresolved.length) {
    dialogs.alert(
      "批量校验：存在未决状态",
      "CSV 本身通过，但本交易日同一 CSV 有未决行：\n" +
      unresolved.map(function (x) { return "#" + x.row + " " + x.status; }).join("\n") +
      "\n\n请先人工核对，不要直接重跑。\n\n报告：\n" + reportPath
    );
    return;
  }

  dialogs.alert(
    "批量校验通过",
    "订单数: " + orders.length +
    "\n买入: " + report.buy_count + " / 卖出: " + report.sell_count +
    "\n计划金额: " + Number(totalNotional).toFixed(2) + " 元" +
    "\n状态来源: " + loaded.source +
    "\n\n该脚本未访问同花顺 UI，也未执行任何委托。" +
    "\n\n报告：\n" + reportPath
  );
}

try {
  run();
} catch (e) {
  console.show();
  console.error(e && e.stack ? e.stack : String(e));
  dialogs.alert("批量校验失败", String(e));
}
