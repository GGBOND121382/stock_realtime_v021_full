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
  orders.forEach(function (order) {
    var sideCode = order.side + ":" + order.code;
    if (seenSideCode[sideCode]) throw new Error("duplicate CSV order: " + sideCode);
    seenSideCode[sideCode] = true;
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

function newState(fingerprint, orders, totalNotional, csvName, sessionDate) {
  return {
    status: "ready",
    session_date: String(sessionDate || ""),
    csv_file: String(csvName || "smoke_orders.csv"),
    csv_fingerprint: String(fingerprint),
    order_count: orders.length,
    total_notional: Number(totalNotional),
    created_at: new Date().toISOString(),
    write_generation: 0,
    results: []
  };
}

function sameScope(state, fingerprint, orders, sessionDate) {
  return !!(
    state &&
    String(state.session_date || "") === String(sessionDate || "") &&
    String(state.csv_fingerprint || "") === String(fingerprint) &&
    Number(state.order_count) === orders.length &&
    Array.isArray(state.results)
  );
}

function restoreState(rawState, fingerprint, orders, totalNotional, csvName, sessionDate) {
  if (sameScope(rawState, fingerprint, orders, sessionDate)) return rawState;
  return newState(fingerprint, orders, totalNotional, csvName, sessionDate);
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

function defaultIo() {
  return {
    exists: function (path) { return files.exists(path); },
    read: function (path) { return files.read(path); },
    write: function (path, text) { files.write(path, text); },
    copy: function (from, to) { return files.copy(from, to); },
    remove: function (path) { return !files.exists(path) || files.remove(path); },
    move: function (from, to) { return files.move(from, to); }
  };
}

function parseStateText(text) {
  var state = JSON.parse(String(text || ""));
  if (!state || typeof state !== "object" || Array.isArray(state)) {
    throw new Error("batch state is not an object");
  }
  if (!Array.isArray(state.results)) throw new Error("batch state results is not an array");
  return state;
}

function readParsed(path, io) {
  if (!io.exists(path)) return { exists: false, valid: false, state: null, error: null };
  try {
    return { exists: true, valid: true, state: parseStateText(io.read(path)), error: null };
  } catch (e) {
    return { exists: true, valid: false, state: null, error: e };
  }
}

function isLegacySameCsv(state, fingerprint, orders) {
  return !!(
    state &&
    !state.session_date &&
    String(state.csv_fingerprint || "") === String(fingerprint) &&
    Number(state.order_count) === orders.length
  );
}

function loadDurable(path, fingerprint, orders, totalNotional, csvName, sessionDate, suppliedIo) {
  var io = suppliedIo || defaultIo();
  var backupPath = path + ".bak";
  var primary = readParsed(path, io);
  var backup = readParsed(backupPath, io);

  if (!primary.exists && !backup.exists) {
    return {
      state: newState(fingerprint, orders, totalNotional, csvName, sessionDate),
      source: "new"
    };
  }

  if (primary.valid && sameScope(primary.state, fingerprint, orders, sessionDate)) {
    return { state: primary.state, source: "primary" };
  }

  if (!primary.exists) {
    if (backup.valid && sameScope(backup.state, fingerprint, orders, sessionDate)) {
      return { state: backup.state, source: "backup" };
    }
    if (backup.exists && !backup.valid) {
      throw new Error("batch state primary is missing and backup is corrupt; refusing automatic replay: " + path);
    }
    if (backup.valid && isLegacySameCsv(backup.state, fingerprint, orders)) {
      throw new Error(
        "legacy batch backup lacks session_date; verify previous orders manually before resetting state: " + backupPath
      );
    }
    return {
      state: newState(fingerprint, orders, totalNotional, csvName, sessionDate),
      source: "new_scope"
    };
  }

  if (!primary.valid) {
    if (backup.valid && sameScope(backup.state, fingerprint, orders, sessionDate)) {
      return { state: backup.state, source: "backup" };
    }
    throw new Error(
      "batch state primary is corrupt and no valid in-scope backup exists; refusing automatic replay: " + path
    );
  }

  if (isLegacySameCsv(primary.state, fingerprint, orders)) {
    throw new Error(
      "legacy batch state lacks session_date; verify previous orders manually before resetting state: " + path
    );
  }

  // A valid state from another date or another CSV belongs to another test session.
  return {
    state: newState(fingerprint, orders, totalNotional, csvName, sessionDate),
    source: "new_scope"
  };
}

function verifyWritten(path, expectedGeneration, io) {
  var parsed = readParsed(path, io);
  if (!parsed.valid) throw new Error("failed to verify written batch state: " + path);
  if (Number(parsed.state.write_generation) !== Number(expectedGeneration)) {
    throw new Error(
      "batch state generation mismatch after write: expected=" + expectedGeneration +
      " actual=" + parsed.state.write_generation
    );
  }
  return parsed.state;
}

function persistDurable(path, state, suppliedIo) {
  var io = suppliedIo || defaultIo();
  var tmpPath = path + ".tmp";
  var backupPath = path + ".bak";
  var nextGeneration = Number(state.write_generation || 0) + 1;
  state.write_generation = nextGeneration;
  state.updated_at = new Date().toISOString();
  var text = JSON.stringify(state, null, 2);

  // Build and verify a complete replacement before touching the primary file.
  io.write(tmpPath, text);
  verifyWritten(tmpPath, nextGeneration, io);

  // Never overwrite a good backup with a corrupt primary.
  var oldPrimary = readParsed(path, io);
  if (oldPrimary.valid) {
    if (!io.copy(path, backupPath)) {
      io.remove(tmpPath);
      throw new Error("failed to backup existing batch state: " + backupPath);
    }
    if (!readParsed(backupPath, io).valid) {
      io.remove(tmpPath);
      throw new Error("backup batch state verification failed: " + backupPath);
    }
  }

  if (io.exists(path) && !io.remove(path)) {
    io.remove(tmpPath);
    throw new Error("failed to remove old batch state before replace: " + path);
  }

  var replaced = io.move(tmpPath, path);
  if (!replaced) {
    if (!io.copy(tmpPath, path)) {
      throw new Error("failed to install new batch state: " + path);
    }
    io.remove(tmpPath);
  }

  verifyWritten(path, nextGeneration, io);
  return state;
}

module.exports = {
  parseCsvLine: parseCsvLine,
  readCsvText: readCsvText,
  normalizeOrder: normalizeOrder,
  validateUniqueOrders: validateUniqueOrders,
  fingerprintText: fingerprintText,
  newState: newState,
  sameScope: sameScope,
  restoreState: restoreState,
  itemForRow: itemForRow,
  isSafeTerminal: isSafeTerminal,
  isUnresolved: isUnresolved,
  unresolvedRows: unresolvedRows,
  parseStateText: parseStateText,
  loadDurable: loadDurable,
  persistDurable: persistDurable
};