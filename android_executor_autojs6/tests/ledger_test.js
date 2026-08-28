"use strict";

var assert = require("assert");
var ledgerPath = require.resolve("../lib/ledger.js");

function loadLedger(storage) {
  delete require.cache[ledgerPath];
  global.storages = {
    create: function () { return storage; }
  };
  return require("../lib/ledger.js");
}

(function testSimpleTerminalStatesUsePutSync() {
  var values = {};
  var asyncCalls = 0;
  var syncCalls = 0;
  var storage = {
    get: function (key, fallback) {
      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : fallback;
    },
    put: function () {
      asyncCalls++;
      throw new Error("async put must never be used");
    },
    putSync: function (key, value) {
      syncCalls++;
      values[key] = value;
    }
  };

  var ledger = loadLedger(storage);
  var order = {
    signal_id: "signal-1234567890abcdef",
    code: "600521",
    side: "buy",
    qty: 300,
    submit_price: 16.06,
    sequence: 1
  };

  ledger.markStarted(order);
  assert.strictEqual(ledger.isTerminal(order.signal_id), false);

  ledger.markResult(order, { success: true }, false);
  assert.strictEqual(ledger.get(order.signal_id).status, "submitted");
  assert.strictEqual(ledger.isTerminal(order.signal_id), true);

  var unknown = new Error("unknown broker result");
  unknown.ambiguous = true;
  ledger.markManualRequired(order, unknown);
  assert.strictEqual(ledger.get(order.signal_id).status, "unknown");
  assert.strictEqual(ledger.isTerminal(order.signal_id), true);

  assert.strictEqual(asyncCalls, 0);
  assert.strictEqual(syncCalls, 3);
})();

(function testOrdinaryFailureIsRetryable() {
  var values = {};
  var storage = {
    get: function (key, fallback) {
      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : fallback;
    },
    putSync: function (key, value) { values[key] = value; }
  };
  var ledger = loadLedger(storage);
  var order = {
    signal_id: "signal-retryable-0001",
    code: "000032",
    side: "buy",
    qty: 300,
    submit_price: 16.06,
    sequence: 1
  };
  ledger.markManualRequired(order, new Error("fill failed"));
  assert.strictEqual(ledger.get(order.signal_id).status, "manual_required");
  assert.strictEqual(ledger.isTerminal(order.signal_id), false);
})();

(function testMissingPutSyncFailsClosed() {
  var storage = {
    get: function (key, fallback) { return fallback; },
    put: function () { throw new Error("must not fall back to async put"); }
  };
  var ledger = loadLedger(storage);
  var thrown = null;
  try {
    ledger.markStarted({
      signal_id: "signal-abcdef1234567890",
      code: "600000",
      side: "buy",
      qty: 100,
      submit_price: 10,
      sequence: 1
    });
  } catch (e) {
    thrown = e;
  }
  assert.ok(thrown);
  assert.strictEqual(thrown.stage, "sync_ledger_unsupported");
  assert.strictEqual(thrown.fatal_ui_state, true);
})();

delete require.cache[ledgerPath];
delete global.storages;

console.log("ledger tests: OK");
