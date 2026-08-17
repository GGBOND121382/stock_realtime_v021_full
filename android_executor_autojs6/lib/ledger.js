"use strict";

var storage = storages.create("as1455_android_executor_v1");

function key(signalId) {
  return "signal:" + String(signalId);
}

function get(signalId) {
  return storage.get(key(signalId), null);
}

function isTerminal(signalId) {
  var item = get(signalId);
  // Only a broker submission is permanently terminal. A dry-run is deliberately
  // repeatable and must never prevent the same signal from being submitted later
  // after the operator explicitly switches config.mode to "live".
  return !!(item && item.status === "submitted");
}

function mark(signalId, payload) {
  var item = payload || {};
  item.signal_id = String(signalId);
  item.updated_at = new Date().toISOString();
  storage.put(key(signalId), item);
  return item;
}

function markStarted(order) {
  return mark(order.signal_id, {
    status: "started",
    code: order.code,
    side: order.side,
    qty: order.qty,
    submit_price: order.submit_price,
    sequence: order.sequence
  });
}

function markResult(order, result, dryRun) {
  return mark(order.signal_id, {
    status: dryRun ? "dry_run" : "submitted",
    code: order.code,
    side: order.side,
    qty: order.qty,
    submit_price: order.submit_price,
    sequence: order.sequence,
    broker_result: result || null
  });
}

function markFailed(order, message) {
  return mark(order.signal_id, {
    status: "failed",
    code: order.code,
    side: order.side,
    qty: order.qty,
    submit_price: order.submit_price,
    sequence: order.sequence,
    error: String(message)
  });
}

module.exports = {
  get: get,
  isTerminal: isTerminal,
  markStarted: markStarted,
  markResult: markResult,
  markFailed: markFailed
};
