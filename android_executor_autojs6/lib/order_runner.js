"use strict";

var ths = require("./ths_adapter.js");

function call(store, name) {
  if (!store || typeof store[name] !== "function") return null;
  var args = Array.prototype.slice.call(arguments, 2);
  return store[name].apply(store, args);
}

function recoveryFailure(order, error) {
  var fatal = new Error(
    "THS failed to recover trading page after order " + order.code + ": " + String(error)
  );
  fatal.stage = "post_order_recovery_failed";
  fatal.ambiguous = false;
  fatal.fatal_ui_state = true;
  return fatal;
}

function execute(order, config, store) {
  call(store, "markStarted", order);

  var hooks = null;
  if (config.mode === "live") {
    hooks = {
      onConfirmationPhaseStarted: function (detail) {
        call(store, "markConfirmationPending", order, detail);
      },
      onConfirmationReady: function (detail) {
        call(store, "markConfirmationOpen", order, detail);
      }
    };
  }

  try {
    var result = config.mode === "live" ?
      ths.submit(order, config, hooks) :
      ths.preview(order, config);

    if (result && result.outcome === "rejected") {
      call(store, "markRejected", order, result);
      return { status: "rejected", result: result };
    }

    call(store, "markResult", order, result, config.mode !== "live");
    return { status: "completed", result: result };
  } catch (e) {
    call(store, "markError", order, e, config.mode !== "live");

    if (e && (e.ambiguous === true || e.fatal_ui_state === true)) {
      throw e;
    }

    try {
      ths.recoverToTradingPage(order.side, config);
    } catch (recoverError) {
      var fatal = recoveryFailure(order, recoverError);
      call(store, "markError", order, fatal, config.mode !== "live");
      throw fatal;
    }

    return {
      status: config.mode === "live" ? "manual_required" : "failed",
      error: e
    };
  }
}

module.exports = {
  execute: execute
};