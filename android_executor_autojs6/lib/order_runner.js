"use strict";

var ths = require("./ths_adapter.js");

function call(store, name) {
  if (!store || typeof store[name] !== "function") return null;
  var args = Array.prototype.slice.call(arguments, 2);
  return store[name].apply(store, args);
}

function persistenceFailure(stage, sourceError, ambiguous) {
  var err = new Error("executor persistence failed at " + stage + ": " + String(sourceError));
  err.stage = stage;
  err.ambiguous = ambiguous === true;
  err.fatal_ui_state = true;
  err.persistence_failure = true;
  return err;
}

function persist(store, name, stage, ambiguous) {
  var args = Array.prototype.slice.call(arguments, 4);
  try {
    return call.apply(null, [store, name].concat(args));
  } catch (e) {
    throw persistenceFailure(stage, e, ambiguous);
  }
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
  // If this write fails no broker UI has been touched yet, but still stop rather
  // than execute an order that cannot be tracked locally.
  persist(store, "markStarted", "persist_started_failed", false, order);

  var hooks = null;
  if (config.mode === "live") {
    hooks = {
      onConfirmationPhaseStarted: function (detail) {
        // This happens before tapping the transaction button. Failure must stop the
        // run; do not continue into a confirmation phase that is not durable.
        persist(
          store,
          "markConfirmationPending",
          "persist_confirmation_pending_failed",
          false,
          order,
          detail
        );
      },
      onConfirmationReady: function (detail) {
        // The broker confirmation dialog is now open. A persistence failure here is
        // ambiguous because the user could still act on that live dialog.
        persist(
          store,
          "markConfirmationOpen",
          "persist_confirmation_open_failed",
          true,
          order,
          detail
        );
      }
    };
  }

  try {
    var result = config.mode === "live" ?
      ths.submit(order, config, hooks) :
      ths.preview(order, config);

    if (result && result.outcome === "rejected") {
      persist(store, "markRejected", "persist_rejected_failed", false, order, result);
      return { status: "rejected", result: result };
    }

    // For a live submitted order, failure to persist the final result must never be
    // converted into a retryable state. confirmation_open/pending remains the last
    // durable state and therefore blocks replay on restart.
    persist(
      store,
      "markResult",
      "persist_final_result_failed",
      config.mode === "live",
      order,
      result,
      config.mode !== "live"
    );
    return { status: "completed", result: result };
  } catch (e) {
    if (e && e.persistence_failure === true) throw e;

    try {
      call(store, "markError", order, e, config.mode !== "live");
    } catch (persistError) {
      throw persistenceFailure(
        "persist_error_state_failed",
        persistError,
        !!(e && e.ambiguous === true)
      );
    }

    if (e && (e.ambiguous === true || e.fatal_ui_state === true)) {
      throw e;
    }

    try {
      ths.recoverToTradingPage(order.side, config);
    } catch (recoverError) {
      var fatal = recoveryFailure(order, recoverError);
      try { call(store, "markError", order, fatal, config.mode !== "live"); } catch (ignore) {}
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