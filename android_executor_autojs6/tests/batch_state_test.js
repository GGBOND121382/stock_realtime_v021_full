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

assert.strictEqual(batch.validateUniqueOrders([
  orders[0],
  { code: "000001", symbol: "000001.SZ", side: "sell", qty: 100, submit_price: 10, sequence: 2 }
]), true);

var fp = batch.fingerprintText(doc.text);
assert.strictEqual(fp, batch.fingerprintText(doc.text));
assert.notStrictEqual(fp, batch.fingerprintText(doc.text + "\n"));

var sessionDate = "2026-08-18";
var state = batch.newState(fp, orders, 1111.5, "smoke_orders.csv", sessionDate);
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

var restored = batch.restoreState(state, fp, orders, 1111.5, "smoke_orders.csv", sessionDate);
assert.strictEqual(restored, state);
var reset = batch.restoreState(state, fp, orders, 1111.5, "smoke_orders.csv", "2026-08-19");
assert.notStrictEqual(reset, state);
assert.strictEqual(reset.results.length, 0);

function memoryIo(initial) {
  var data = Object.assign({}, initial || {});
  return {
    data: data,
    exists: function (path) { return Object.prototype.hasOwnProperty.call(data, path); },
    read: function (path) {
      if (!Object.prototype.hasOwnProperty.call(data, path)) throw new Error("missing " + path);
      return data[path];
    },
    write: function (path, text) { data[path] = String(text); },
    copy: function (from, to) {
      if (!Object.prototype.hasOwnProperty.call(data, from)) return false;
      data[to] = data[from];
      return true;
    },
    remove: function (path) {
      if (!Object.prototype.hasOwnProperty.call(data, path)) return true;
      delete data[path];
      return true;
    },
    move: function (from, to) {
      if (!Object.prototype.hasOwnProperty.call(data, from)) return false;
      data[to] = data[from];
      delete data[from];
      return true;
    }
  };
}

var statePath = "/batch.json";
var io = memoryIo();
var fresh = batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, io);
assert.strictEqual(fresh.source, "new");
fresh.state.results.push({ row: 1, status: "submitted" });
batch.persistDurable(statePath, fresh.state, io);
assert.strictEqual(batch.parseStateText(io.data[statePath]).results[0].status, "submitted");
assert.strictEqual(batch.parseStateText(io.data[statePath]).write_generation, 1);

fresh.state.results.push({ row: 2, status: "confirmation_open" });
batch.persistDurable(statePath, fresh.state, io);
assert.strictEqual(batch.parseStateText(io.data[statePath]).write_generation, 2);
assert.strictEqual(batch.parseStateText(io.data[statePath + ".bak"]).write_generation, 1);

// Corrupt primary recovers from current-session backup.
io.data[statePath] = "{broken";
var recovered = batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, io);
assert.strictEqual(recovered.source, "backup");
assert.strictEqual(recovered.state.results[0].status, "submitted");

// Crash window after primary removal but before tmp installation: backup-only must
// also recover, never reset to a fresh batch.
var ioBackupOnly = memoryIo();
ioBackupOnly.data[statePath + ".bak"] = JSON.stringify(recovered.state);
var backupOnly = batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, ioBackupOnly);
assert.strictEqual(backupOnly.source, "backup");
assert.strictEqual(backupOnly.state.results[0].status, "submitted");

// Missing primary + corrupt backup fails closed.
ioBackupOnly.data[statePath + ".bak"] = "{broken-backup";
assert.throws(function () {
  batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, ioBackupOnly);
}, /refusing automatic replay/);

// Corrupt both primary and backup fails closed.
io.data[statePath + ".bak"] = "{also-broken";
assert.throws(function () {
  batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, io);
}, /refusing automatic replay/);

var prior = batch.newState(fp, orders, 1111.5, "smoke_orders.csv", "2026-08-17");
prior.results.push({ row: 1, status: "submitted" });
var ioPrior = memoryIo((function () {
  var x = {};
  x[statePath] = JSON.stringify(prior);
  return x;
})());
var newDay = batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, ioPrior);
assert.strictEqual(newDay.source, "new_scope");
assert.strictEqual(newDay.state.results.length, 0);

var legacy = batch.newState(fp, orders, 1111.5, "smoke_orders.csv", "");
delete legacy.session_date;
var ioLegacy = memoryIo((function () {
  var x = {};
  x[statePath] = JSON.stringify(legacy);
  return x;
})());
assert.throws(function () {
  batch.loadDurable(statePath, fp, orders, 1111.5, "smoke_orders.csv", sessionDate, ioLegacy);
}, /legacy batch state lacks session_date/);

console.log("batch_state tests: OK");