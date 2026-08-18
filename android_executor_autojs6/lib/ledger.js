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
  if (!item) return false;
  // These states must never be replayed automatically. confirmation_* means the
  // previous process entered the broker-confirmation phase but did not persist a
  // final outcome before it ended.
  return [
    "submitted",
    "rejected",
    "unknown",
    "confirmation_pending",
    "confirmation_open"
  ].indexOf(String(item.status || "")) >= 0;
}

function mark(signalId, payload) {
  var item = payload || {};
  item.signal_id = String(signalId);
  item.updated_at = new Date().toISOString();
  storage.put(key(signalId), item);
  return item;
}

function baseOrder(order) {
  return {
    code: order.code,
    side: order.side,
    qty: order.qty,
    submit_price: order.submit_price,
    sequence: order.sequence
  };
}

function markStarted(order) {
  var item = baseOrder(order);
  item.status = "started";
  return mark(order.signal_id, item);
}

function markConfirmationPending(order, detail) {
  var item = baseOrder(order);
  item.status = "confirmation_pending";
  item.confirmation = detail || null;
  return mark(order.signal_id, item);
}

function markConfirmationOpen(order, detail) {
  var item = baseOrder(order);
  item.status = "confirmation_open";
  item.confirmation = detail || null;
  return mark(order.signal_id, item);
}

function markResult(order, result, dryRun) {
  var item = baseOrder(order);
  item.status = dryRun ? "dry_run" : "submitted";
  item.broker_result = result || null;
  return mark(order.signal_id, item);
}

function markRejected(order, result) {
  var item = baseOrder(order);
  item.status = "rejected";
  item.broker_result = result || null;
  return mark(order.signal_id, item);
}

function markManualRequired(order, error) {
  var item = baseOrder(order);
  var ambiguous = !!(error && error.ambiguous === true);
  item.status = ambiguous ? "unknown" : "manual_required";
  item.stage = error && error.stage ? String(error.stage) : "manual_required";
  item.ambiguous = ambiguous;
  item.error = String(error);
  return mark(order.signal_id, item);
}

function markFailed(order, message) {
  var item = baseOrder(order);
  item.status = "failed";
  item.error = String(message);
  return mark(order.signal_id, item);
}

module.exports = {
  get: get,
  isTerminal: isTerminal,
  markStarted: markStarted,
  markConfirmationPending: markConfirmationPending,
  markConfirmationOpen: markConfirmationOpen,
  markResult: markResult,
  markRejected: markRejected,
  markManualRequired: markManualRequired,
  markFailed: markFailed
};