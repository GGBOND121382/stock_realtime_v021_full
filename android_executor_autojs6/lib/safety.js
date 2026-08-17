"use strict";

function fail(message) {
  throw new Error("SAFETY: " + message);
}

function todayShanghai() {
  // Android device should be set to China Standard Time for production use.
  return new java.text.SimpleDateFormat("yyyy-MM-dd").format(new java.util.Date());
}

function positiveInteger(value, name) {
  var n = Number(value);
  if (!isFinite(n) || n <= 0 || Math.floor(n) !== n) {
    fail(name + " must be a positive integer: " + value);
  }
  return n;
}

function positivePrice(value, name) {
  var n = Number(value);
  if (!isFinite(n) || n <= 0) {
    fail(name + " must be a positive number: " + value);
  }
  return n;
}

function validateOrder(order, experiment) {
  if (!order || typeof order !== "object") fail("invalid order object");
  if (!order.signal_id || String(order.signal_id).length < 16) fail("missing/short signal_id");
  var side = String(order.side || "").toLowerCase();
  if (side !== "buy" && side !== "sell") fail("invalid side: " + side);
  var code = String(order.code || "");
  if (!/^\d{6}$/.test(code)) fail("invalid code: " + code);
  var qty = positiveInteger(order.qty, "qty");
  if (side === "buy" && qty % 100 !== 0) fail("BUY qty must be a multiple of 100: " + qty);
  positivePrice(order.submit_price, "submit_price");
  positiveInteger(order.sequence, "sequence");
  return {
    signal_id: String(order.signal_id),
    code: code,
    symbol: String(order.symbol || code),
    side: side,
    qty: qty,
    submit_price: Number(order.submit_price),
    sequence: Number(order.sequence),
    experiment: experiment
  };
}

function validateBatch(batch, config) {
  if (!batch || typeof batch !== "object") fail("empty batch");
  if (String(batch.status || "").toLowerCase() !== "ready") fail("batch is not READY");
  if (String(batch.protocol || "") !== "as1455_execution_batch_v1") fail("unexpected protocol: " + batch.protocol);
  if (String(batch.experiment || "") !== String(config.production_experiment)) fail("unexpected experiment: " + batch.experiment);
  var today = todayShanghai();
  if (String(batch.trade_date || "") !== today) fail("trade_date=" + batch.trade_date + " today=" + today);
  if (!Array.isArray(batch.orders)) fail("orders is not an array");
  if (Number(batch.order_count) !== batch.orders.length) fail("order_count mismatch");
  if (batch.orders.length > Number(config.max_orders || 60)) fail("too many orders: " + batch.orders.length);

  var seenSignal = {};
  var seenSequence = {};
  var orders = batch.orders.map(function (order) {
    var normalized = validateOrder(order, batch.experiment);
    if (seenSignal[normalized.signal_id]) fail("duplicate signal_id: " + normalized.signal_id);
    if (seenSequence[normalized.sequence]) fail("duplicate sequence: " + normalized.sequence);
    seenSignal[normalized.signal_id] = true;
    seenSequence[normalized.sequence] = true;
    return normalized;
  });
  orders.sort(function (a, b) { return a.sequence - b.sequence; });
  for (var i = 0; i < orders.length; i++) {
    if (orders[i].sequence !== i + 1) fail("sequence must be contiguous from 1");
  }
  var seenBuy = false;
  orders.forEach(function (o) {
    if (o.side === "buy") seenBuy = true;
    if (o.side === "sell" && seenBuy) fail("sell order appears after buy order");
  });
  return {
    status: "ready",
    protocol: String(batch.protocol),
    trade_date: String(batch.trade_date),
    experiment: String(batch.experiment),
    orders: orders
  };
}

module.exports = {
  validateBatch: validateBatch,
  todayShanghai: todayShanghai
};
