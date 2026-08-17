"use strict";
var assert = require("assert");
var q = require("../lib/manual_order_queue.js");

var csv = [
  "symbol,side,shares,raw_exec_price,name",
  "000032.SZ,buy,300,16.06,测试A",
  "601615.SH,BUY,500,9.80,测试B",
  "600000.SH,sell,37,12.34,测试C"
].join("\n");

var orders = q.parseOrdersCsv(csv, {});
assert.strictEqual(orders.length, 3);
assert.strictEqual(orders[0].code, "000032");
assert.strictEqual(orders[0].market, "SZ");
assert.strictEqual(orders[0].qty, 300);
assert.strictEqual(orders[0].price_text, "16.06");
assert.strictEqual(orders[2].side, "sell");
assert.strictEqual(orders[2].qty, 37);

assert.throws(function () {
  q.parseOrdersCsv("symbol,side,shares\n000001.SZ,buy,150", {});
}, /multiple of 100/);

assert.throws(function () {
  q.parseOrdersCsv("symbol,side,shares\n000001.SZ,buy,100\n000001.SZ,buy,200", {});
}, /duplicate order/);

var fp = q.fnv1a(csv);
assert.strictEqual(fp, q.fnv1a(csv));
var state = q.createState(fp, orders);
assert.deepStrictEqual(q.counts(state, orders), { total: 3, pending: 3, done: 0, skipped: 0 });
q.markDone(state, orders[0]);
q.markSkipped(state, orders[1]);
assert.deepStrictEqual(q.counts(state, orders), { total: 3, pending: 1, done: 1, skipped: 1 });
assert.strictEqual(q.findNextIndex(state, orders, "normal", 0), 2);
assert.strictEqual(q.findNextIndex(state, orders, "skipped", 0), 1);
q.reopen(state, orders[1]);
assert.strictEqual(q.counts(state, orders).pending, 2);

console.log("manual_order_queue tests: OK");
