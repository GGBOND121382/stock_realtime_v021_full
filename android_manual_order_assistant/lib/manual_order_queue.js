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
  if (quoted) throw new Error("CSV contains an unterminated quoted field");
  fields.push(current);
  return fields;
}

function cleanLines(text) {
  return String(text || "")
    .replace(/^\uFEFF/, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter(function (line) { return line.trim().length > 0; });
}

function findHeader(headers, names) {
  for (var i = 0; i < names.length; i++) {
    var index = headers.indexOf(names[i]);
    if (index >= 0) return index;
  }
  return -1;
}

function positiveInteger(raw, label) {
  var value = Number(raw);
  if (!isFinite(value) || value <= 0 || Math.floor(value) !== value) {
    throw new Error(label + " must be a positive integer: " + raw);
  }
  return value;
}

function normalizePrice(raw, label) {
  var text = String(raw === null || raw === undefined ? "" : raw).trim();
  if (!text) return "";
  var value = Number(text);
  if (!isFinite(value) || value <= 0) throw new Error(label + " must be positive: " + text);
  return text;
}

function normalizeSymbol(raw, rowNumber) {
  var symbol = String(raw || "").trim().toUpperCase();
  var match = symbol.match(/^(\d{6})(?:\.(SZ|SH))?$/);
  if (!match) throw new Error("CSV row " + rowNumber + " invalid symbol: " + symbol);
  return { symbol: symbol, code: match[1], market: match[2] || "" };
}

function parseOrdersCsv(text, options) {
  options = options || {};
  var lines = cleanLines(text);
  if (lines.length < 2) throw new Error("CSV has no data rows");

  var headers = parseCsvLine(lines[0]).map(function (x) { return String(x).trim(); });
  var symbolIndex = findHeader(headers, ["symbol", "code", "证券代码", "股票代码"]);
  var sideIndex = findHeader(headers, ["side", "方向", "买卖"]);
  var qtyIndex = findHeader(headers, ["shares", "qty", "quantity", "数量", "股数"]);
  var priceIndex = findHeader(headers, ["raw_exec_price", "submit_price", "price", "价格", "委托价"]);
  var nameIndex = findHeader(headers, ["name", "stock_name", "security_name", "证券名称", "股票名称"]);

  if (symbolIndex < 0) throw new Error("CSV missing symbol/code column");
  if (sideIndex < 0) throw new Error("CSV missing side column");
  if (qtyIndex < 0) throw new Error("CSV missing shares/qty column");

  var rows = [];
  var duplicates = {};
  for (var i = 1; i < lines.length; i++) {
    var values = parseCsvLine(lines[i]);
    var rowNumber = i + 1;
    var parsedSymbol = normalizeSymbol(values[symbolIndex], rowNumber);
    var side = String(values[sideIndex] || "").trim().toLowerCase();
    if (side === "b" || side === "buy" || side === "买" || side === "买入") side = "buy";
    else if (side === "s" || side === "sell" || side === "卖" || side === "卖出") side = "sell";
    else throw new Error("CSV row " + rowNumber + " invalid side: " + values[sideIndex]);

    var qty = positiveInteger(values[qtyIndex], "CSV row " + rowNumber + " quantity");
    if (side === "buy" && qty % 100 !== 0) {
      throw new Error("CSV row " + rowNumber + " BUY quantity must be a multiple of 100 shares: " + qty);
    }

    var price = priceIndex >= 0 ? normalizePrice(values[priceIndex], "CSV row " + rowNumber + " price") : "";
    var name = nameIndex >= 0 ? String(values[nameIndex] || "").trim() : "";
    var duplicateKey = side + ":" + parsedSymbol.code;
    if (duplicates[duplicateKey] && options.allow_duplicates !== true) {
      throw new Error("duplicate order in CSV: " + duplicateKey + " (rows " + duplicates[duplicateKey] + " and " + rowNumber + ")");
    }
    duplicates[duplicateKey] = rowNumber;

    rows.push({
      id: "row-" + rowNumber,
      row_number: rowNumber,
      sequence: rows.length + 1,
      symbol: parsedSymbol.symbol,
      code: parsedSymbol.code,
      market: parsedSymbol.market,
      name: name,
      side: side,
      qty: qty,
      price_text: price
    });
  }
  return rows;
}

function fnv1a(text) {
  var hash = 2166136261;
  var s = String(text || "");
  for (var i = 0; i < s.length; i++) {
    hash ^= s.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return (hash >>> 0).toString(16);
}

function createState(fingerprint, orders) {
  var status = {};
  for (var i = 0; i < orders.length; i++) status[orders[i].id] = "pending";
  return {
    version: 1,
    fingerprint: String(fingerprint),
    created_at: new Date().toISOString(),
    started_at: null,
    updated_at: null,
    completed_at: null,
    cursor: 0,
    review_cursor: 0,
    phase: "normal",
    status: status,
    history: []
  };
}

function sanitizeState(state, fingerprint, orders) {
  if (!state || state.version !== 1 || String(state.fingerprint) !== String(fingerprint)) {
    return createState(fingerprint, orders);
  }
  var valid = {};
  for (var i = 0; i < orders.length; i++) valid[orders[i].id] = true;
  if (!state.status || typeof state.status !== "object") state.status = {};
  Object.keys(state.status).forEach(function (key) {
    if (!valid[key]) delete state.status[key];
  });
  for (var j = 0; j < orders.length; j++) {
    var id = orders[j].id;
    if (["pending", "done", "skipped"].indexOf(state.status[id]) < 0) state.status[id] = "pending";
  }
  if (!Array.isArray(state.history)) state.history = [];
  if (typeof state.cursor !== "number") state.cursor = 0;
  if (typeof state.review_cursor !== "number") state.review_cursor = 0;
  if (state.phase !== "normal" && state.phase !== "skipped") state.phase = "normal";
  return state;
}

function counts(state, orders) {
  var result = { total: orders.length, pending: 0, done: 0, skipped: 0 };
  for (var i = 0; i < orders.length; i++) {
    var st = state.status[orders[i].id] || "pending";
    if (st === "done") result.done++;
    else if (st === "skipped") result.skipped++;
    else result.pending++;
  }
  return result;
}

function findNextIndex(state, orders, phase, startIndex) {
  var wanted = phase === "skipped" ? "skipped" : "pending";
  var start = Math.max(0, Number(startIndex) || 0);
  for (var i = start; i < orders.length; i++) {
    if ((state.status[orders[i].id] || "pending") === wanted) return i;
  }
  // Skipped-item review is one-way: never wrap to an earlier skipped order.
  if (phase === "skipped") return -1;
  for (var j = 0; j < start; j++) {
    if ((state.status[orders[j].id] || "pending") === wanted) return j;
  }
  return -1;
}

function record(state, order, action) {
  var entry = { at: new Date().toISOString(), order_id: order.id, action: action };
  state.history.push(entry);
  if (state.history.length > 200) state.history = state.history.slice(state.history.length - 200);
  state.updated_at = entry.at;
}

function markDone(state, order) {
  state.status[order.id] = "done";
  record(state, order, "done");
}

function markSkipped(state, order) {
  state.status[order.id] = "skipped";
  record(state, order, "skipped");
}

function reopen(state, order) {
  state.status[order.id] = "pending";
  record(state, order, "reopened");
}

module.exports = {
  parseCsvLine: parseCsvLine,
  parseOrdersCsv: parseOrdersCsv,
  fnv1a: fnv1a,
  createState: createState,
  sanitizeState: sanitizeState,
  counts: counts,
  findNextIndex: findNextIndex,
  markDone: markDone,
  markSkipped: markSkipped,
  reopen: reopen
};
