"use strict";

var orderContract = require("./order_contract.js");

var IDS = {
  navButton: "com.hexin.plat.android:id/btn",
  stockCode: "com.hexin.plat.android:id/auto_stockcode",
  searchContainer: "com.hexin.plat.android:id/dialogplus_view_container",
  stockSuggestionCode: "com.hexin.plat.android:id/stockcode_tv",
  stockVolume: "com.hexin.plat.android:id/stockvolume",
  stockPrice: "com.hexin.plat.android:id/stockprice",
  transaction: "com.hexin.plat.android:id/btn_transaction",
  dialogLayout: "com.hexin.plat.android:id/dialog_layout",
  dialogTitle: "com.hexin.plat.android:id/dialog_title",
  promptContent: "com.hexin.plat.android:id/prompt_content",
  okButton: "com.hexin.plat.android:id/ok_btn",
  cancelButton: "com.hexin.plat.android:id/cancel_btn"
};

var EDIT_TEXT = "android.widget.EditText";
var TEXT_VIEW = "android.widget.TextView";
var THS_PACKAGE = "com.hexin.plat.android";
var TAP_ATTEMPTS = 3;
var PROBE_MAX_NODES = 400;

function timeout(config) {
  return Number((config && config.ui_timeout_ms) || 5000);
}

function fillTimeout(config) {
  var configured = Number(config && config.fill_timeout_ms);
  if (isFinite(configured) && configured > 0) return configured;
  return Math.min(timeout(config), 1800);
}

function fieldVerifyTimeout(config) {
  var configured = Number(config && config.field_verify_timeout_ms);
  return isFinite(configured) && configured > 0 ? configured : 700;
}

function manualConfirmTimeout(config) {
  var configured = Number(config && config.manual_confirm_timeout_ms);
  return isFinite(configured) && configured > 0 ? configured : 5500;
}

function manualResultGrace(config) {
  var configured = Number(config && config.manual_result_grace_ms);
  return isFinite(configured) && configured > 0 ? configured : 1000;
}

function waitId(resourceId, timeoutMs) {
  var node = id(resourceId).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS UI node not found: " + resourceId);
  return node;
}

function clickNode(node) {
  if (!node) return false;
  try {
    if (node.click()) return true;
  } catch (e) {}
  try {
    var b = node.bounds();
    return click(b.centerX(), b.centerY());
  } catch (e2) {
    return false;
  }
}

function tapTarget(node) {
  var cur = node;
  for (var i = 0; i < 5 && cur; i++) {
    try {
      if (cur.clickable()) return cur;
      cur = cur.parent();
    } catch (e) {
      break;
    }
  }
  return node;
}

function tapCenter(node) {
  var target = tapTarget(node);
  if (!target) return false;
  try {
    var b = target.bounds();
    return click(b.centerX(), b.centerY());
  } catch (e) {
    return false;
  }
}

function safeNodeText(node) {
  try {
    return node ? String(node.text() || "") : "";
  } catch (e) {
    return "";
  }
}

function safeNodeClass(node) {
  try {
    return node ? String(node.className() || "") : "";
  } catch (e) {
    return "";
  }
}

function nodeBoundsKey(node) {
  try {
    var b = node.bounds();
    return [b.left, b.top, b.right, b.bottom].join(":");
  } catch (e) {
    return "";
  }
}

function topologyError(message) {
  var err = new Error("THS field topology invalid: " + message);
  err.stage = "field_topology_invalid";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  return err;
}

function crossWriteError(message) {
  var err = new Error("THS cross-field write detected: " + message);
  err.stage = "field_cross_write_detected";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  return err;
}

function dialogContractError(message, stage) {
  var err = new Error("THS confirmation contract invalid: " + message);
  err.stage = stage || "confirmation_contract_invalid";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  return err;
}

function unknownUiError(message, stage) {
  var err = new Error(message);
  err.stage = stage || "unknown_ui_state";
  err.ambiguous = true;
  err.fatal_ui_state = true;
  return err;
}

function setNodeTextOnce(node, value, label) {
  var v = String(value);
  if (!node || safeNodeClass(node) !== EDIT_TEXT) {
    throw topologyError(label + " is not an EditText");
  }

  // Exactly one write. Do not use global setText(), do not click-and-write again,
  // and do not retry the whole order automatically. Android accessibility text
  // updates can be asynchronous; callers verify the resulting state separately.
  try {
    node.setText(v);
  } catch (e) {
    var err = new Error("THS node-scoped setText failed: " + label + ": " + String(e));
    err.stage = "field_write_failed";
    err.ambiguous = false;
    throw err;
  }
}

function findInside(container, selector, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var node = null;
    try { node = container.findOne(selector); } catch (e) { node = null; }
    if (node) return node;
    sleep(60);
  }
  return null;
}

function findInsideOnce(container, selector) {
  try { return container ? container.findOne(selector) : null; } catch (e) { return null; }
}

function waitExactEditable(resourceId, label, timeoutMs) {
  var t = Number(timeoutMs || 5000);
  var container = waitId(resourceId, t);
  if (safeNodeClass(container) === EDIT_TEXT) return container;

  var edit = findInside(container, className(EDIT_TEXT), t);
  if (!edit) {
    throw topologyError(label + " EditText not found under " + resourceId);
  }
  return edit;
}

function assertDistinctFields(codeEdit, volumeEdit, priceEdit) {
  var fields = [
    { label: "code", node: codeEdit },
    { label: "volume", node: volumeEdit },
    { label: "price", node: priceEdit }
  ];
  var seenBounds = {};

  for (var i = 0; i < fields.length; i++) {
    var item = fields[i];
    if (!item.node || safeNodeClass(item.node) !== EDIT_TEXT) {
      throw topologyError(item.label + " node is missing or not EditText");
    }
    var bounds = nodeBoundsKey(item.node);
    if (!bounds) throw topologyError(item.label + " has no readable bounds");
    if (seenBounds[bounds]) {
      throw topologyError(
        item.label + " resolves to same bounds as " + seenBounds[bounds] + " (" + bounds + ")"
      );
    }
    seenBounds[bounds] = item.label;
  }
}

function normalizeNumericText(value) {
  var text = String(value === null || value === undefined ? "" : value)
    .replace(/,/g, "")
    .trim();
  var n = Number(text);
  return isFinite(n) ? n : null;
}

function fieldMatches(actual, expected, numeric) {
  if (!numeric) return String(actual) === String(expected);
  var a = normalizeNumericText(actual);
  var e = normalizeNumericText(expected);
  return a !== null && e !== null && Math.abs(a - e) < 0.000001;
}

function waitNodeValue(node, expected, timeoutMs, numeric) {
  var deadline = Date.now() + Number(timeoutMs || 700);
  while (Date.now() <= deadline) {
    if (fieldMatches(safeNodeText(node), expected, numeric === true)) return true;
    sleep(60);
  }
  return false;
}

function clickTradeTab(label, timeoutMs) {
  var node = id(IDS.navButton).text(label).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS trading tab not found: " + label);
  if (!clickNode(node)) throw new Error("THS trading tab click failed: " + label);
}

function ensureTradingPage(side, config) {
  var pkg = String(config.ths_package || THS_PACKAGE);
  if (currentPackage() !== pkg) {
    app.launchPackage(pkg);
    sleep(650);
  }
  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(1000);
  if (!codeEdit) {
    clickTradeTab(side === "sell" ? "卖出" : "买入", timeout(config));
    codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(timeout(config));
  }
  if (!codeEdit) throw new Error("not on THS trading page");
  return codeEdit;
}

function recoverToTradingPage(side, config) {
  try {
    if (id(IDS.searchContainer).findOnce()) {
      back();
      sleep(120);
    }
  } catch (e) {}
  ensureTradingPage(side, config);
  clickTradeTab(side === "sell" ? "卖出" : "买入", timeout(config));
  sleep(140);
  return true;
}

function findSuggestionOnce(searchContainer, code) {
  try {
    var scoped = searchContainer.findOne(id(IDS.stockSuggestionCode).text(String(code)));
    if (scoped) return scoped;
    return searchContainer.findOne(className(TEXT_VIEW).text(String(code)));
  } catch (e) {
    return null;
  }
}

function isMainTradeCodeResolved(code) {
  if (id(IDS.searchContainer).findOnce()) return false;
  var codeNode = id(IDS.stockCode).className(EDIT_TEXT).text(String(code)).findOnce();
  var transaction = id(IDS.transaction).findOnce();
  return !!(codeNode && transaction);
}

function waitMainTradeCodeResolved(code, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 900);
  while (Date.now() <= deadline) {
    if (isMainTradeCodeResolved(code)) return true;
    sleep(60);
  }
  return false;
}

function fillCode(code, config) {
  var expected = String(code);
  if (isMainTradeCodeResolved(expected)) return;

  var t = fillTimeout(config);
  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(t);
  if (!codeEdit) throw new Error("THS stock code EditText not found");
  if (!clickNode(codeEdit)) throw new Error("THS stock code click failed");
  sleep(120);

  var searchContainer = waitId(IDS.searchContainer, t);
  var searchEdit = findInside(searchContainer, id(IDS.stockCode).className(EDIT_TEXT), t);
  if (!searchEdit) throw topologyError("stock-code search EditText not found inside search container");
  setNodeTextOnce(searchEdit, expected, "stock-code search");

  var deadline = Date.now() + t;
  while (Date.now() <= deadline) {
    if (isMainTradeCodeResolved(expected)) return;

    var activeSearch = id(IDS.searchContainer).findOnce();
    if (activeSearch) {
      var suggestion = findSuggestionOnce(activeSearch, expected);
      if (suggestion) {
        for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
          if (!tapCenter(suggestion)) break;
          if (waitMainTradeCodeResolved(expected, 500)) return;
          activeSearch = id(IDS.searchContainer).findOnce();
          if (!activeSearch) break;
          suggestion = findSuggestionOnce(activeSearch, expected);
          if (!suggestion) break;
        }
      }
    }
    sleep(70);
  }

  if (isMainTradeCodeResolved(expected)) return;
  var err = new Error("THS stock code not resolved on final trade form code=" + expected);
  err.stage = "stock_code_not_resolved";
  err.ambiguous = false;
  throw err;
}

function fillOrderFieldsOnce(order, config) {
  ensureTradingPage(order.side, config);
  clickTradeTab(order.side === "sell" ? "卖出" : "买入", timeout(config));
  sleep(140);
  fillCode(order.code, config);

  if (!isMainTradeCodeResolved(String(order.code))) {
    var codeErr = new Error("THS final code verification failed code=" + order.code);
    codeErr.stage = "stock_code_final_verify_failed";
    throw codeErr;
  }

  var qtyText = String(order.qty);
  var priceText = Number(order.submit_price).toFixed(2);
  var verifyMs = fieldVerifyTimeout(config);

  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(fillTimeout(config));
  var volumeEdit = waitExactEditable(IDS.stockVolume, "quantity", fillTimeout(config));
  var priceEdit = waitExactEditable(IDS.stockPrice, "price", fillTimeout(config));
  assertDistinctFields(codeEdit, volumeEdit, priceEdit);

  setNodeTextOnce(volumeEdit, qtyText, "quantity");
  if (!waitNodeValue(volumeEdit, qtyText, verifyMs, true)) {
    var qtyErr = new Error("THS quantity verification failed expected=" + qtyText + " actual=" + safeNodeText(volumeEdit));
    qtyErr.stage = "quantity_verify_failed";
    throw qtyErr;
  }
  if (safeNodeText(codeEdit) !== String(order.code)) {
    throw crossWriteError("quantity write changed code expected=" + order.code + " actual=" + safeNodeText(codeEdit));
  }

  setNodeTextOnce(priceEdit, priceText, "price");
  if (!waitNodeValue(priceEdit, priceText, verifyMs, true)) {
    var priceErr = new Error("THS price verification failed expected=" + priceText + " actual=" + safeNodeText(priceEdit));
    priceErr.stage = "price_verify_failed";
    throw priceErr;
  }

  if (safeNodeText(codeEdit) !== String(order.code)) {
    throw crossWriteError("price write changed code expected=" + order.code + " actual=" + safeNodeText(codeEdit));
  }
  if (!fieldMatches(safeNodeText(volumeEdit), qtyText, true)) {
    throw crossWriteError("price write changed quantity expected=" + qtyText + " actual=" + safeNodeText(volumeEdit));
  }

  return {
    success: true,
    stage: "fields_filled_and_verified",
    code: String(order.code),
    side: String(order.side),
    qty: Number(order.qty),
    submit_price: Number(order.submit_price),
    fill_attempts: 1,
    field_bounds: {
      code: nodeBoundsKey(codeEdit),
      quantity: nodeBoundsKey(volumeEdit),
      price: nodeBoundsKey(priceEdit)
    }
  };
}

function fillOrderFields(order, config) {
  try {
    return fillOrderFieldsOnce(order, config);
  } catch (e) {
    if (!e.stage) e.stage = "fill_failed";
    if (e.ambiguous !== true) e.ambiguous = false;
    try {
      e.probe = dumpProbe(order, e.stage);
      e.message += "; probe=" + e.probe;
    } catch (probeError) {}
    throw e;
  }
}

function safeCollectionTexts(container) {
  var texts = [];
  if (!container) return texts;
  try {
    var nodes = container.find(className(TEXT_VIEW));
    for (var i = 0; i < nodes.size(); i++) {
      var value = safeNodeText(nodes.get(i));
      if (value) texts.push(value);
    }
  } catch (e) {}
  return texts;
}

function dialogSnapshot() {
  var dialog = id(IDS.dialogLayout).findOnce();
  if (!dialog) return null;

  var titleNode = findInsideOnce(dialog, id(IDS.dialogTitle)) || id(IDS.dialogTitle).findOnce();
  var promptNode = findInsideOnce(dialog, id(IDS.promptContent)) || id(IDS.promptContent).findOnce();
  var okNode = findInsideOnce(dialog, id(IDS.okButton)) || id(IDS.okButton).findOnce();
  var cancelNode = findInsideOnce(dialog, id(IDS.cancelButton)) || id(IDS.cancelButton).findOnce();
  var texts = safeCollectionTexts(dialog);

  return {
    dialog: dialog,
    title: safeNodeText(titleNode),
    message: safeNodeText(promptNode),
    ok: okNode,
    cancel: cancelNode,
    texts: texts
  };
}

function confirmationLabels(side) {
  return side === "sell" ? { title: "委托卖出确认", action: "确认卖出" } :
    { title: "委托买入确认", action: "确认买入" };
}

function findConfirmation(order) {
  var snap = dialogSnapshot();
  if (!snap) return null;
  var parsed = orderContract.parseConfirmationTexts(snap.texts);
  var expectedTitle = confirmationLabels(order.side).title;
  if (String(parsed.title || "").indexOf(expectedTitle) < 0) return null;
  return {
    dialog: snap.dialog,
    parsed: parsed,
    ok: snap.ok,
    cancel: snap.cancel,
    labels: confirmationLabels(order.side)
  };
}

function findResultDialog() {
  var snap = dialogSnapshot();
  if (!snap || !snap.message) return null;
  var parsedConfirmation = orderContract.parseConfirmationTexts(snap.texts);
  if (parsedConfirmation.title) return null;
  var classified = orderContract.classifyResultMessage(snap.message);
  return {
    dialog: snap.dialog,
    title: snap.title || "系统信息",
    message: snap.message,
    outcome: classified.outcome,
    ok: snap.ok
  };
}

function findUnknownDialog(order) {
  var snap = dialogSnapshot();
  if (!snap) return null;
  if (findConfirmation(order) || findResultDialog()) return null;
  return snap;
}

function validateConfirmationOrCancel(order, confirmation) {
  var check = orderContract.validateConfirmation(order, confirmation.parsed);
  if (check.ok) return confirmation;

  var cancelled = cancelOpenConfirmation(order, confirmation);
  var err = dialogContractError(check.errors.join("; "), "confirmation_contract_mismatch");
  if (!cancelled) {
    err.message += "; confirmation could not be cancelled";
  }
  throw err;
}

function waitTransactionOutcome(order, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 1500);
  while (Date.now() <= deadline) {
    var confirmation = findConfirmation(order);
    if (confirmation) return { kind: "confirmation", confirmation: confirmation };

    var result = findResultDialog();
    if (result) return { kind: "result", result: result };

    var unknownDialog = findUnknownDialog(order);
    if (unknownDialog) return { kind: "unknown_dialog", dialog: unknownDialog };
    sleep(70);
  }
  return null;
}

function extractContractNumber(message) {
  var m = String(message || "").match(/合同号(?:为)?[：:\s]*([0-9\s]+)/);
  return m ? String(m[1]).replace(/\s/g, "") : "";
}

function dismissResultDialog(order, result) {
  if (!result || !result.ok) {
    throw unknownUiError("THS result dialog has no exact OK button", "result_dialog_missing_ok");
  }
  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    if (!id(IDS.dialogLayout).findOnce()) return;
    if (!tapCenter(result.ok)) break;
    sleep(220);
    if (!id(IDS.dialogLayout).findOnce()) return;
    result = findResultDialog() || result;
  }
  var probe = dumpProbe(order, "result_dialog_not_dismissed_after_retries");
  throw unknownUiError(
    "THS result dialog did not dismiss after retries; probe=" + probe,
    "result_dialog_stuck"
  );
}

function cancelOpenConfirmation(order, supplied) {
  var current = supplied || findConfirmation(order);
  if (!current) return true;
  if (!current.cancel) return false;
  for (var attempt = 1; attempt <= 2; attempt++) {
    if (!tapCenter(current.cancel)) break;
    sleep(180);
    if (!findConfirmation(order)) return true;
    current = findConfirmation(order);
    if (!current || !current.cancel) return !current;
  }
  return !findConfirmation(order);
}

function fireHook(hooks, name, payload) {
  if (!hooks || typeof hooks[name] !== "function") return;
  hooks[name](payload);
}

function safeValue(fn, fallback) {
  try {
    var v = fn();
    return v === null || v === undefined ? fallback : v;
  } catch (e) { return fallback; }
}

function clean(v) {
  return String(v === null || v === undefined ? "" : v).replace(/\r/g, "\\r").replace(/\n/g, "\\n");
}

function timestampToken() {
  var d = new Date();
  function pad(n) { return n < 10 ? "0" + n : String(n); }
  return d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "_" +
    pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
}

function dumpProbe(order, stage) {
  var lines = [
    "========== THS UI PROBE ==========",
    "stage=" + String(stage),
    "order=" + JSON.stringify(order || {}),
    "currentPackage=" + safeValue(function () { return currentPackage(); }, ""),
    "currentActivity=" + safeValue(function () { return currentActivity(); }, ""),
    "---------- nodes ----------"
  ];
  var nodes = null;
  try { nodes = classNameMatches(/.*/).find(); } catch (e) { lines.push("probe_error=" + String(e)); }
  if (nodes) {
    var limit = Math.min(nodes.size(), PROBE_MAX_NODES);
    for (var i = 0; i < limit; i++) {
      var n = nodes.get(i);
      var pkg = safeValue(function () { return n.packageName(); }, "");
      var rid = safeValue(function () { return n.id(); }, "");
      var txt = safeValue(function () { return n.text(); }, "");
      var desc = safeValue(function () { return n.desc(); }, "");
      if (String(pkg) === THS_PACKAGE || rid || txt || desc) {
        var b = safeValue(function () { return n.bounds(); }, null);
        lines.push("[" + i + "] pkg=" + clean(pkg) + " id=" + clean(rid) +
          " text=" + JSON.stringify(clean(txt)) + " desc=" + JSON.stringify(clean(desc)) +
          " class=" + clean(safeValue(function () { return n.className(); }, "")) +
          " clickable=" + safeValue(function () { return n.clickable(); }, false) +
          " bounds=" + (b ? clean(b.toString()) : ""));
      }
    }
  }
  lines.push("========== END THS UI PROBE ==========");
  var path = files.join(files.cwd(), "ths_order_confirmation_probe_" + timestampToken() + ".txt");
  files.write(path, lines.join("\n"));
  return path;
}

function openOrderConfirmation(order, config, hooks) {
  var fillResult = fillOrderFields(order, config);
  var waitMs = Math.max(700, Math.min(1500, Math.floor(timeout(config) / 3)));
  var phaseMarked = false;

  function markPhase() {
    if (phaseMarked) return;
    phaseMarked = true;
    fireHook(hooks, "onConfirmationPhaseStarted", {
      stage: "confirmation_phase_started",
      fill: fillResult
    });
  }

  var existing = findConfirmation(order);
  if (existing) {
    markPhase();
    existing = validateConfirmationOrCancel(order, existing);
    fireHook(hooks, "onConfirmationReady", existing.parsed);
    return { kind: "confirmation", confirmation: existing, fill: fillResult };
  }

  markPhase();
  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    var submitNode = waitId(IDS.transaction, Math.min(timeout(config), 1200));
    if (!tapCenter(submitNode)) throw new Error("THS transaction button tap failed");

    var outcome = waitTransactionOutcome(order, waitMs);
    if (outcome && outcome.kind === "confirmation") {
      var checked = validateConfirmationOrCancel(order, outcome.confirmation);
      fireHook(hooks, "onConfirmationReady", checked.parsed);
      return { kind: "confirmation", confirmation: checked, fill: fillResult };
    }
    if (outcome && outcome.kind === "result") {
      if (outcome.result.outcome === "rejected") {
        dismissResultDialog(order, outcome.result);
        return { kind: "rejected", result: outcome.result, fill: fillResult };
      }
      throw unknownUiError(
        "THS returned an unrecognized result before manual confirmation: " + outcome.result.message,
        "pre_confirmation_result_unknown"
      );
    }
    if (outcome && outcome.kind === "unknown_dialog") {
      var probeUnknown = dumpProbe(order, "unknown_dialog_after_transaction_tap");
      throw dialogContractError(
        "unrecognized dialog opened after transaction tap; probe=" + probeUnknown,
        "transaction_unknown_dialog"
      );
    }
    sleep(100);
  }

  var probe = dumpProbe(order, "transaction_no_confirmation_after_retries");
  var err = new Error(
    "THS transaction tap did not open a recognized confirmation/result dialog; probe=" + probe
  );
  err.stage = "transaction_no_confirmation";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  throw err;
}

function resultReturn(order, result, opened) {
  if (result.outcome === "unknown") {
    throw unknownUiError(
      "THS returned an unrecognized post-confirmation result: " + result.message,
      "manual_confirmation_result_unknown"
    );
  }

  var returnValue = {
    success: result.outcome === "submitted",
    outcome: result.outcome,
    mode: "live",
    prompt: { title: result.title, content: result.message },
    contract_no: result.outcome === "submitted" ? extractContractNumber(result.message) : "",
    stage: result.outcome === "submitted" ? "submitted_after_manual_confirmation" : "rejected_after_manual_confirmation",
    fill_attempts: opened.fill.fill_attempts
  };
  dismissResultDialog(order, result);
  sleep(120);
  return returnValue;
}

function preview(order, config) {
  var opened = openOrderConfirmation(order, config, null);
  if (opened.kind === "rejected") {
    return {
      success: false,
      outcome: "rejected",
      mode: "dry_run",
      prompt: { title: opened.result.title, content: opened.result.message },
      stage: "rejected_before_confirmation"
    };
  }

  if (!cancelOpenConfirmation(order, opened.confirmation)) {
    var probe = dumpProbe(order, "cancel_confirmation_not_dismissed");
    var err = new Error("THS confirmation cancel did not dismiss; probe=" + probe);
    err.stage = "dry_run_cancel_failed";
    err.ambiguous = false;
    err.fatal_ui_state = true;
    throw err;
  }
  return {
    success: true,
    outcome: "preview_cancelled",
    mode: "dry_run",
    stage: "confirmation_contract_verified_and_cancelled",
    confirmation: opened.confirmation.parsed
  };
}

function submit(order, config, hooks) {
  var opened = openOrderConfirmation(order, config, hooks || null);
  if (opened.kind === "rejected") {
    return {
      success: false,
      outcome: "rejected",
      mode: "live",
      prompt: { title: opened.result.title, content: opened.result.message },
      stage: "rejected_before_manual_confirmation",
      fill_attempts: opened.fill.fill_attempts
    };
  }

  var labels = confirmationLabels(order.side);
  toast("核对后手动点击" + labels.action + "：" + order.code + " x" + order.qty);

  var deadline = Date.now() + manualConfirmTimeout(config);
  while (Date.now() <= deadline) {
    var confirmation = findConfirmation(order);
    if (confirmation) {
      sleep(70);
      continue;
    }

    var result = null;
    var graceDeadline = Date.now() + manualResultGrace(config);
    while (Date.now() <= graceDeadline) {
      result = findResultDialog();
      if (result) break;
      if (findUnknownDialog(order)) {
        var probeUnknown = dumpProbe(order, "unknown_dialog_after_manual_confirmation");
        throw unknownUiError(
          "unrecognized dialog after manual confirmation; probe=" + probeUnknown,
          "manual_confirmation_unknown_dialog"
        );
      }
      sleep(70);
    }
    if (result) return resultReturn(order, result, opened);

    throw unknownUiError(
      "THS manual confirmation dialog disappeared but no recognized result was found",
      "manual_confirmation_result_unrecognized"
    );
  }

  var stillOpen = findConfirmation(order);
  if (stillOpen) {
    throw unknownUiError(
      "THS manual confirmation timed out while confirmation dialog is still open; manual takeover required",
      "manual_confirmation_timeout_open_dialog"
    );
  }

  var lateDeadline = Date.now() + manualResultGrace(config);
  while (Date.now() <= lateDeadline) {
    var lateResult = findResultDialog();
    if (lateResult) return resultReturn(order, lateResult, opened);
    sleep(70);
  }

  throw unknownUiError(
    "THS manual confirmation ended in an unrecognized state",
    "manual_confirmation_state_unknown"
  );
}

module.exports = {
  IDS: IDS,
  preview: preview,
  submit: submit,
  fillOrderFields: fillOrderFields,
  ensureTradingPage: ensureTradingPage,
  recoverToTradingPage: recoverToTradingPage
};
