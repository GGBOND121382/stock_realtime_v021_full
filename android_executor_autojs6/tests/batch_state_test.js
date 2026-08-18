"use strict";

var assert = require("assert");
var batch = require("../lib/batch_state.js");

var csv = [
  "symbol,side,shares,raw_exec_price",
  "000001.SZ,buy,100,10.25",
  "600000.SH,sell,13,8.50"
].join("\n");

var doc = batch.readCsvText(csv);
assert.strictEqual(doc.rows.length, 2);
var orders = doc.rows.map(function (row, i) { return batch.normalizeOrder(row, i + 1); });
assert.strictEqual(orders[0].code, "000001");
assert.strictEqual(orders[0].qty, 100);
assert.strictEqual(orders[1].side, "sell");
assert.strictEqual(orders[1].qty, 13);
assert.strictEqual(batch.validateUniqueOrders(orders), true);

assert.throws(function () {
  batch.readCsvText('symbol,side,shares,raw_exec_price\n"000001.SZ,buy,100,10.25');
}, /unterminated quoted CSV field/);

assert.throws(function () {
  batch.normalizeOrder({ symbol: "000001.SZ", side: "buy", shares: "101", raw_exec_price: "10" }, 1);
}, /BUY shares must be multiple of 100/);

assert.throws(function () {
  batch.validateUniqueOrders([orders[0], orders[0]]);
}, /duplicate CSV order/);

assert.throws(function () {
  batch.validateUniqueOrders([
    orders[0],
    { code: "000001", symbol: "000001.SZ", side: "sell", qty: 100, submit_price: 10, sequence: 2 }
  ]);
}, /both buy and sell sides/);

var fp = batch.fingerprintText(doc.text);
assert.strictEqual(fp, batch.fingerprintText(doc.text));
assert.notStrictEqual(fp, batch.fingerprintText(doc.text + "\n"));

var state = batch.newState(fp, orders, 1111.5, "smoke_orders.csv");
var item1 = batch.itemForRow(state, 1, orders[0]);
assert.strictEqual(item1.status, "pending");
item1.status = "submitted";
assert.strictEqual(batch.itemForRow(state, 1, orders[0]), item1);
assert.strictEqual(batch.isSafeTerminal("submitted"), true);
assert.strictEqual(batch.isSafeTerminal("rejected"), true);
assert.strictEqual(batch.isSafeTerminal("manual_required"), false);

var item2 = batch.itemForRow(state, 2, orders[1]);
item2.status = "confirmation_open";
assert.strictEqual(batch.isUnresolved(item2.status), true);
assert.deepStrictEqual(batch.unresolvedRows(state).map(function (x) { return x.row; }), [2]);

var restored = batch.restoreState(state, fp, orders, 1111.5, "smoke_orders.csv");
assert.strictEqual(restored, state);
var reset = batch.restoreState(state, "different", orders, 1111.5, "smoke_orders.csv");
assert.notStrictEqual(reset, state);
assert.strictEqual(reset.results.length, 0);

console.log("batch_state tests: OK");
