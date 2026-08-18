"use strict";

var assert = require("assert");
var adapterPath = require.resolve("../lib/ths_adapter.js");
var runnerPath = require.resolve("../lib/order_runner.js");

function loadRunner(adapter) {
  delete require.cache[runnerPath];
  require.cache[adapterPath] = {
    id: adapterPath,
    filename: adapterPath,
    loaded: true,
    exports: adapter
  };
  return require("../lib/order_runner.js");
}

function order() {
  return { code: "600521", side: "buy", qty: 300, submit_price: 16.06, sequence: 1 };
}

function makeStore(log, overrides) {
  overrides = overrides || {};
  function fn(name) {
    return function () {
      log.push(name);
      if (overrides[name]) return overrides[name].apply(null, arguments);
    };
  }
  return {
    markStarted: fn("started"),
    markConfirmationPending: fn("confirmation_pending"),
    markConfirmationOpen: fn("confirmation_open"),
    markResult: fn("result"),
    markRejected: fn("rejected"),
    markError: fn("error")
  };
}

// Successful live path must establish both replay barriers before final result.
(function () {
  var log = [];
  var recovered = 0;
  var adapter = {
    submit: function (o, config, hooks) {
      hooks.onConfirmationPhaseStarted({ stage: "pending" });
      hooks.onConfirmationReady({ code: o.code, qty: o.qty, price: o.submit_price });
      return { outcome: "submitted", stage: "submitted_after_manual_confirmation" };
    },
    preview: function () { throw new Error("not used"); },
    recoverToTradingPage: function () { recovered++; }
  };
  var runner = loadRunner(adapter);
  var result = runner.execute(order(), { mode: "live" }, makeStore(log));
  assert.strictEqual(result.status, "completed");
  assert.deepStrictEqual(log, ["started", "confirmation_pending", "confirmation_open", "result"]);
  assert.strictEqual(recovered, 0);
})();

// Explicit broker rejection is terminal, not UNKNOWN/manual retry.
(function () {
  var log = [];
  var adapter = {
    submit: function (o, config, hooks) {
      hooks.onConfirmationPhaseStarted({});
      return { outcome: "rejected", stage: "rejected_before_manual_confirmation" };
    },
    preview: function () {},
    recoverToTradingPage: function () { throw new Error("must not recover rejected result"); }
  };
  var runner = loadRunner(adapter);
  var result = runner.execute(order(), { mode: "live" }, makeStore(log));
  assert.strictEqual(result.status, "rejected");
  assert.deepStrictEqual(log, ["started", "confirmation_pending", "rejected"]);
})();

// Ordinary pre-confirmation fill error may recover and become manual_required.
(function () {
  var log = [];
  var recovered = 0;
  var adapter = {
    submit: function () {
      var e = new Error("fill failed");
      e.stage = "fill_failed_after_retry";
      e.ambiguous = false;
      throw e;
    },
    preview: function () {},
    recoverToTradingPage: function () { recovered++; }
  };
  var runner = loadRunner(adapter);
  var result = runner.execute(order(), { mode: "live" }, makeStore(log));
  assert.strictEqual(result.status, "manual_required");
  assert.deepStrictEqual(log, ["started", "error"]);
  assert.strictEqual(recovered, 1);
})();

// Ambiguous broker state must stop and must not invoke recovery/retry logic.
(function () {
  var log = [];
  var recovered = 0;
  var adapter = {
    submit: function (o, config, hooks) {
      hooks.onConfirmationPhaseStarted({});
      hooks.onConfirmationReady({});
      var e = new Error("result unknown");
      e.stage = "manual_confirmation_result_unrecognized";
      e.ambiguous = true;
      e.fatal_ui_state = true;
      throw e;
    },
    preview: function () {},
    recoverToTradingPage: function () { recovered++; }
  };
  var runner = loadRunner(adapter);
  assert.throws(function () {
    runner.execute(order(), { mode: "live" }, makeStore(log));
  }, /result unknown/);
  assert.deepStrictEqual(log, ["started", "confirmation_pending", "confirmation_open", "error"]);
  assert.strictEqual(recovered, 0);
})();

// Broker success followed by local final-result persistence failure must be fatal
// and ambiguous. It must never recover or downgrade to a retryable state.
(function () {
  var log = [];
  var recovered = 0;
  var adapter = {
    submit: function (o, config, hooks) {
      hooks.onConfirmationPhaseStarted({});
      hooks.onConfirmationReady({});
      return { outcome: "submitted", stage: "submitted_after_manual_confirmation" };
    },
    preview: function () {},
    recoverToTradingPage: function () { recovered++; }
  };
  var runner = loadRunner(adapter);
  var store = makeStore(log, {
    result: function () { throw new Error("disk full"); }
  });
  var thrown = null;
  try {
    runner.execute(order(), { mode: "live" }, store);
  } catch (e) {
    thrown = e;
  }
  assert.ok(thrown);
  assert.strictEqual(thrown.stage, "persist_final_result_failed");
  assert.strictEqual(thrown.ambiguous, true);
  assert.strictEqual(thrown.fatal_ui_state, true);
  assert.strictEqual(recovered, 0);
  assert.deepStrictEqual(log, ["started", "confirmation_pending", "confirmation_open", "result"]);
})();

// Once the live confirmation is visible, failure to persist that barrier is also
// ambiguous and fatal; the runner cannot safely infer whether the user will confirm.
(function () {
  var log = [];
  var recovered = 0;
  var adapter = {
    submit: function (o, config, hooks) {
      hooks.onConfirmationPhaseStarted({});
      hooks.onConfirmationReady({});
      return { outcome: "submitted" };
    },
    preview: function () {},
    recoverToTradingPage: function () { recovered++; }
  };
  var runner = loadRunner(adapter);
  var store = makeStore(log, {
    confirmation_open: function () { throw new Error("storage unavailable"); }
  });
  var thrown = null;
  try {
    runner.execute(order(), { mode: "live" }, store);
  } catch (e) {
    thrown = e;
  }
  assert.ok(thrown);
  assert.strictEqual(thrown.stage, "persist_confirmation_open_failed");
  assert.strictEqual(thrown.ambiguous, true);
  assert.strictEqual(thrown.fatal_ui_state, true);
  assert.strictEqual(recovered, 0);
  assert.deepStrictEqual(log, ["started", "confirmation_pending", "confirmation_open"]);
})();

// Restore actual adapter module for any test process that continues afterwards.
delete require.cache[runnerPath];
delete require.cache[adapterPath];

console.log("order_runner tests: OK");