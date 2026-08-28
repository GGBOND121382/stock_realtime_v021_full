"use strict";

var FILE_NAME = "retry_orders.json";
var VERSION = 1;

function path() {
  return files.join(files.cwd(), FILE_NAME);
}

function emptyState() {
  return {
    version: VERSION,
    updated_at: new Date().toISOString(),
    orders: []
  };
}

function load() {
  var p = path();
  if (!files.exists(p)) return emptyState();
  var raw = JSON.parse(files.read(p));
  if (!raw || !Array.isArray(raw.orders)) throw new Error("invalid " + FILE_NAME);
  raw.version = VERSION;
  return raw;
}

function save(state) {
  var next = state || emptyState();
  next.version = VERSION;
  next.updated_at = new Date().toISOString();
  if (!Array.isArray(next.orders)) next.orders = [];
  files.write(path(), JSON.stringify(next, null, 2));
  return next;
}

function baseOrder(order) {
  return {
    signal_id: String(order.signal_id),
    sequence: Number(order.sequence),
    code: String(order.code),
    symbol: String(order.symbol || order.code),
    side: String(order.side),
    qty: Number(order.qty),
    submit_price: Number(order.submit_price),
    experiment: String(order.experiment || "")
  };
}

function findIndex(state, signalId) {
  var wanted = String(signalId);
  for (var i = 0; i < state.orders.length; i++) {
    if (String(state.orders[i].signal_id) === wanted) return i;
  }
  return -1;
}

function recordFailure(order, error, source) {
  var state = load();
  var idx = findIndex(state, order.signal_id);
  var previous = idx >= 0 ? state.orders[idx] : null;
  var item = baseOrder(order);
  item.stage = error && error.stage ? String(error.stage) : "unknown";
  item.ambiguous = !!(error && error.ambiguous === true);
  item.retryable = !item.ambiguous && !(error && error.fatal_ui_state === true);
  item.error = String(error);
  item.source = String(source || "main");
  item.failure_count = Number(previous && previous.failure_count || 0) + 1;
  item.last_failed_at = new Date().toISOString();
  if (previous && previous.first_failed_at) item.first_failed_at = previous.first_failed_at;
  else item.first_failed_at = item.last_failed_at;

  if (idx >= 0) state.orders[idx] = item;
  else state.orders.push(item);
  state.orders.sort(function (a, b) { return Number(a.sequence) - Number(b.sequence); });
  save(state);
  return item;
}

function recordFromLedger(order, ledgerItem) {
  var state = load();
  if (findIndex(state, order.signal_id) >= 0) return state;
  var item = baseOrder(order);
  var status = String(ledgerItem && ledgerItem.status || "");
  item.stage = String(ledgerItem && ledgerItem.stage || (status === "started" ? "interrupted_after_start" : "unknown"));
  item.ambiguous = status === "unknown" || status === "started";
  item.retryable = status === "manual_required" || status === "failed";
  item.error = String(ledgerItem && ledgerItem.error || (status ? "ledger status=" + status : ""));
  item.source = "ledger_bootstrap";
  item.failure_count = 1;
  item.first_failed_at = String(ledgerItem && ledgerItem.updated_at || new Date().toISOString());
  item.last_failed_at = item.first_failed_at;
  state.orders.push(item);
  state.orders.sort(function (a, b) { return Number(a.sequence) - Number(b.sequence); });
  return save(state);
}

function remove(signalId) {
  var state = load();
  var wanted = String(signalId);
  state.orders = state.orders.filter(function (item) {
    return String(item.signal_id) !== wanted;
  });
  return save(state);
}

module.exports = {
  FILE_NAME: FILE_NAME,
  path: path,
  load: load,
  save: save,
  recordFailure: recordFailure,
  recordFromLedger: recordFromLedger,
  remove: remove
};
