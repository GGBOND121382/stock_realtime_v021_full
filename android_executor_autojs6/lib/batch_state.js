"use strict";

function parseCsvLine(line) {
  var fields = [];
  var current = "";
  var quoted = false;
  for (var i = 0; i < line.length; i++) {
    var ch = line.charAt(i);
    if (ch === '"') {
      if (quoted && i + 1 < line.length && line.charAt(i + 1) === '"') {
        current += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === "," && !quoted) {
      fields.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  if (quoted) throw new Error("unterminated quoted CSV field");
  fields.push(current);
  return fields;
}

function readCsvText(text) {
  var normalized = String(text || "")
    .replace(/^\uFEFF/, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n");
  var lines = normalized.split("\n").filter(function (line) { return line.trim().length > 0; });
  if (lines.length < 2) throw new Error("batch CSV has no data rows");

  var headers = parseCsvLine(lines[0]).map(function (x) { return x.trim(); });
  var required = ["symbol", "side", "shares", "raw_exec_price"];
  required.forEach(function (name) {
    if (headers.indexOf(name) < 0) throw new Error("batch CSV missing column: " + name);
  });

  var rows = [];
  for (var i = 1; i < lines.length; i++) {
    var values = parseCsvLine(lines[i]);
    if (values.length > headers.length) throw new Error("CSV row " + i + " has too many columns");
    var row = {};
    for (var j = 0; j < headers.length; j++) {
      row[headers[j]] = values[j] === undefined ? "" : values[j];
    }
    rows.push(row);
  }
  return { text: normalized, headers: headers, rows: rows };
}

function normalizeOrder(row, rowNumber) {
  var symbol = String(row.symbol || "").trim();
  var match = symbol.match(/^(\d{6})(?:\.(?:SZ|SH))?$/i);
  if (!match) throw new Error("CSV row " + rowNumber + " invalid symbol: " + symbol);

  var side = String(row.side || "").trim().toLowerCase();
  if (side !== "buy" && side !== "sell") throw new Error("CSV row " + rowNumber + " invalid side: " + side);

  var qty = Number(row.shares);
  if (!isFinite(qty) || qty <= 0 || Math.floor(qty) !== qty) {
    throw new Error("CSV row " + rowNumber + " invalid shares: " + row.shares);
  }
  if (side === "buy" && qty % 100 !== 0) {
    throw new Error("CSV row " + rowNumber + " BUY shares must be multiple of 100: " + qty);
  }

  var price = Number(row.raw_exec_price);
  if (!isFinite(price) || price <= 0) {
    throw new Error("CSV row " + rowNumber + " invalid raw_exec_price: " + row.raw_exec_price);
  }

  return {
    code: match[1],
    symbol: symbol,
    side: side,
    qty: qty,
    submit_price: price,
    sequence: rowNumber
  };
}

function validateUniqueOrders(orders) {
  var seenSideCode = {};
  var seenCode = {};
  orders.forEach(function (order) {
    var sideCode = order.side + ":" + order.code;
    if (seenSideCode[sideCode]) throw new Error("duplicate CSV order: " + sideCode);
    if (seenCode[order.code] && seenCode[order.code] !== order.side) {
      throw new Error("same code appears on both buy and sell sides: " + order.code);
    }
    seenSideCode[sideCode] = true;
    seenCode[order.code] = order.side;
  });
  return true;
}

function fingerprintText(text) {
  var hash = 2166136261;
  var value = String(text || "");
  for (var i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return (hash >>> 0).toString(16);
}

function newState(fingerprint, orders, totalNotional, csvName) {
  return {
    status: "ready",
    csv_file: String(csvName || "smoke_orders.csv"),
    csv_fingerprint: fingerprint,
    order_count: orders.length,
    total_notional: totalNotional,
    created_at: new Date().toISOString(),
    results: []
  };
}

function restoreState(rawState, fingerprint, orders, totalNotional, csvName) {
  if (rawState && rawState.csv_fingerprint === fingerprint &&
      Number(rawState.order_count) === orders.length && Array.isArray(rawState.results)) {
    return rawState;
  }
  return newState(fingerprint, orders, totalNotional, csvName);
}

function itemForRow(state, row, order) {
  for (var i = 0; i < state.results.length; i++) {
    if (Number(state.results[i].row) === Number(row)) return state.results[i];
  }
  var item = { row: row, order: order, status: "pending", attempts: 0 };
  state.results.push(item);
  return item;
}

function isSafeTerminal(status) {
  return status === "submitted" || status === "rejected";
}

function isUnresolved(status) {
  return ["confirmation_pending", "confirmation_open", "unknown", "blocked"].indexOf(String(status || "")) >= 0;
}

function unresolvedRows(state) {
  return state.results.filter(function (item) { return isUnresolved(item.status); });
}

module.exports = {
  parseCsvLine: parseCsvLine,
  readCsvText: readCsvText,
  normalizeOrder: normalizeOrder,
  validateUniqueOrders: validateUniqueOrders,
  fingerprintText: fingerprintText,
  newState: newState,
  restoreState: restoreState,
  itemForRow: itemForRow,
  isSafeTerminal: isSafeTerminal,
  isUnresolved: isUnresolved,
  unresolvedRows: unresolvedRows
};
