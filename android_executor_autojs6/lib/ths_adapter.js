"use strict";

var IDS = {
  navButton: "com.hexin.plat.android:id/btn",
  stockCode: "com.hexin.plat.android:id/auto_stockcode",
  searchContainer: "com.hexin.plat.android:id/dialogplus_view_container",
  stockSuggestionCode: "com.hexin.plat.android:id/stockcode_tv",
  stockVolume: "com.hexin.plat.android:id/stockvolume",
  stockPrice: "com.hexin.plat.android:id/stockprice",
  transaction: "com.hexin.plat.android:id/btn_transaction"
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

function setNodeTextOnce(node, value, label) {
  if (!node) throw new Error("THS input missing: " + label);
  try {
    node.setText(String(value));
  } catch (e) {
    var err = new Error("THS setText failed: " + label + ": " + String(e));
    err.stage = label + "_write_failed";
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

function waitDescendantEdit(containerId, label, timeoutMs) {
  var t = Number(timeoutMs || 5000);
  var container = waitId(containerId, t);
  var edit = findInside(container, className(EDIT_TEXT), t);
  if (!edit) {
    var err = new Error("THS EditText not found under " + label + ": " + containerId);
    err.stage = label + "_edit_missing";
    err.ambiguous = false;
    throw err;
  }
  return edit;
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

function waitFieldValue(containerId, expected, timeoutMs, numeric) {
  var deadline = Date.now() + Number(timeoutMs || 700);
  while (Date.now() <= deadline) {
    try {
      var container = id(containerId).findOnce();
      if (container) {
        var edit = container.findOne(className(EDIT_TEXT));
        if (edit && fieldMatches(safeNodeText(edit), expected, numeric === true)) return true;
      }
    } catch (e) {}
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
    var node = searchContainer.findOne(id(IDS.stockSuggestionCode).text(String(code)));
    if (node) return node;
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
  if (!searchEdit) throw new Error("THS stock search EditText not found");

  // Write exactly once. The search overlay may replace this UiObject immediately,
  // so do not use the old node's text() as a success/failure signal.
  setNodeTextOnce(searchEdit, expected, "stock_code");

  var deadline = Date.now() + t;
  var clickedSuggestion = false;
  while (Date.now() <= deadline) {
    if (isMainTradeCodeResolved(expected)) return;

    var activeSearch = id(IDS.searchContainer).findOnce();
    if (activeSearch && !clickedSuggestion) {
      var suggestion = findSuggestionOnce(activeSearch, expected);
      if (suggestion) {
        clickedSuggestion = true;
        if (!tapCenter(suggestion)) {
          var tapErr = new Error("THS stock suggestion tap failed code=" + expected);
          tapErr.stage = "stock_suggestion_tap_failed";
          tapErr.ambiguous = false;
          throw tapErr;
        }
        if (waitMainTradeCodeResolved(expected, Math.min(900, t))) return;
        break;
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

function fillOrderFields(order, config) {
  ensureTradingPage(order.side, config);
  clickTradeTab(order.side === "sell" ? "卖出" : "买入", timeout(config));
  sleep(140);
  fillCode(order.code, config);

  if (!isMainTradeCodeResolved(String(order.code))) {
    var codeErr = new Error("THS final code verification failed code=" + order.code);
    codeErr.stage = "stock_code_final_verify_failed";
    codeErr.ambiguous = false;
    throw codeErr;
  }

  var qtyText = String(order.qty);
  var priceText = Number(order.submit_price).toFixed(2);
  var verifyMs = fieldVerifyTimeout(config);

  var volumeEdit = waitDescendantEdit(IDS.stockVolume, "quantity", fillTimeout(config));
  setNodeTextOnce(volumeEdit, qtyText, "quantity");
  if (!waitFieldValue(IDS.stockVolume, qtyText, verifyMs, true)) {
    var qtyErr = new Error("THS quantity verification failed expected=" + qtyText);
    qtyErr.stage = "quantity_verify_failed";
    qtyErr.ambiguous = false;
    throw qtyErr;
  }

  var priceEdit = waitDescendantEdit(IDS.stockPrice, "price", fillTimeout(config));
  setNodeTextOnce(priceEdit, priceText, "price");
  if (!waitFieldValue(IDS.stockPrice, priceText, verifyMs, true)) {
    var priceErr = new Error("THS price verification failed expected=" + priceText);
    priceErr.stage = "price_verify_failed";
    priceErr.ambiguous = false;
    throw priceErr;
  }

  return {
    success: true,
    stage: "fields_filled_and_verified",
    code: String(order.code),
    side: String(order.side),
    qty: Number(order.qty),
    submit_price: Number(order.submit_price),
    fill_attempts: 1
  };
}

function confirmationLabels(side) {
  return side === "sell" ? { title: "委托卖出确认", action: "确认卖出" } :
    { title: "委托买入确认", action: "确认买入" };
}

function findConfirmation(order) {
  var labels = confirmationLabels(order.side);
  var titleNode = text(labels.title).findOnce();
  var actionNode = text(labels.action).findOnce();
  var cancelNode = text("取消").findOnce();
  if (!titleNode || !actionNode || !cancelNode) return null;
  return { title: titleNode, action: actionNode, cancel: cancelNode, labels: labels };
}

function waitConfirmation(order, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var c = findConfirmation(order);
    if (c) return c;
    sleep(70);
  }
  return null;
}

function extractContractNumber(message) {
  var m = String(message || "").match(/合同号(?:为)?[：:\s]*([0-9\s]+)/);
  return m ? String(m[1]).replace(/\s/g, "") : "";
}

function findSuccessDialog() {
  var messageNode = textContains("委托已提交").findOnce();
  var okNode = text("确定").findOnce();
  if (!messageNode || !okNode) return null;
  var titleNode = text("系统信息").findOnce();
  var message = String(messageNode.text() || "");
  return {
    title: titleNode ? String(titleNode.text() || "") : "系统信息",
    message: message,
    contract_no: extractContractNumber(message),
    ok: okNode
  };
}

function waitSuccessDialog(timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var s = findSuccessDialog();
    if (s) return s;
    sleep(70);
  }
  return null;
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

function openOrderConfirmation(order, config) {
  var fillResult = fillOrderFields(order, config);
  var waitMs = Math.max(700, Math.min(1300, Math.floor(timeout(config) / 3)));

  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    var existing = findConfirmation(order);
    if (existing) return { confirmation: existing, fill: fillResult };

    var submitNode = waitId(IDS.transaction, Math.min(timeout(config), 1200));
    if (!tapCenter(submitNode)) throw new Error("THS transaction button tap failed");

    var confirmation = waitConfirmation(order, waitMs);
    if (confirmation) return { confirmation: confirmation, fill: fillResult };
    sleep(100);
  }

  var probe = dumpProbe(order, "transaction_no_confirmation_after_retries");
  var err = new Error(
    "THS transaction tap did not open " + confirmationLabels(order.side).title + "; probe=" + probe
  );
  err.stage = "transaction_no_confirmation";
  err.ambiguous = false;
  throw err;
}

function dismissSuccessDialog(order, dialog) {
  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    if (!textContains("委托已提交").findOnce()) return;
    var current = findSuccessDialog() || dialog;
    if (!current || !current.ok || !tapCenter(current.ok)) break;
    sleep(220);
    if (!textContains("委托已提交").findOnce()) return;
  }
  var probe = dumpProbe(order, "success_dialog_not_dismissed_after_retries");
  var err = new Error("THS success dialog did not dismiss after retries; probe=" + probe);
  err.stage = "success_dialog_stuck";
  err.ambiguous = true;
  err.fatal_ui_state = true;
  throw err;
}

function cancelOpenConfirmation(order) {
  var current = findConfirmation(order);
  if (!current) return true;
  for (var attempt = 1; attempt <= 2; attempt++) {
    if (!tapCenter(current.cancel)) break;
    sleep(180);
    if (!findConfirmation(order)) return true;
    current = findConfirmation(order);
    if (!current) return true;
  }
  return !findConfirmation(order);
}

function preview(order, config) {
  openOrderConfirmation(order, config);
  if (!cancelOpenConfirmation(order)) {
    var probe = dumpProbe(order, "cancel_confirmation_not_dismissed");
    var err = new Error("THS confirmation cancel did not dismiss; probe=" + probe);
    err.stage = "dry_run_cancel_failed";
    err.ambiguous = false;
    err.fatal_ui_state = true;
    throw err;
  }
  return { success: true, mode: "dry_run", stage: "confirmation_reached_and_cancelled" };
}

function submit(order, config) {
  var opened = openOrderConfirmation(order, config);
  var labels = confirmationLabels(order.side);
  toast("核对后手动点击" + labels.action + "：" + order.code + " x" + order.qty);

  var deadline = Date.now() + manualConfirmTimeout(config);
  while (Date.now() <= deadline) {
    var success = findSuccessDialog();
    if (success) {
      dismissSuccessDialog(order, success);
      sleep(120);
      return {
        success: true,
        mode: "live",
        prompt: { title: success.title, content: success.message },
        contract_no: success.contract_no,
        stage: "submitted_after_manual_confirmation",
        fill_attempts: opened.fill.fill_attempts
      };
    }

    if (!findConfirmation(order)) {
      success = waitSuccessDialog(manualResultGrace(config));
      if (success) {
        dismissSuccessDialog(order, success);
        sleep(120);
        return {
          success: true,
          mode: "live",
          prompt: { title: success.title, content: success.message },
          contract_no: success.contract_no,
          stage: "submitted_after_manual_confirmation",
          fill_attempts: opened.fill.fill_attempts
        };
      }

      var ambiguous = new Error(
        "THS manual confirmation dialog disappeared but no recognized submission result was found"
      );
      ambiguous.stage = "manual_confirmation_result_unrecognized";
      ambiguous.ambiguous = true;
      throw ambiguous;
    }
    sleep(70);
  }

  // Do not click Cancel at the timeout boundary. The user may be confirming at the
  // same moment; leave the dialog untouched and require manual takeover instead.
  if (findConfirmation(order)) {
    var timeoutError = new Error(
      "THS manual confirmation timed out while confirmation dialog is still open"
    );
    timeoutError.stage = "manual_confirmation_timeout_open_dialog";
    timeoutError.ambiguous = true;
    throw timeoutError;
  }

  var lateSuccess = waitSuccessDialog(manualResultGrace(config));
  if (lateSuccess) {
    dismissSuccessDialog(order, lateSuccess);
    return {
      success: true,
      mode: "live",
      prompt: { title: lateSuccess.title, content: lateSuccess.message },
      contract_no: lateSuccess.contract_no,
      stage: "submitted_after_manual_confirmation",
      fill_attempts: opened.fill.fill_attempts
    };
  }

  var unknown = new Error("THS manual confirmation ended in an unrecognized state");
  unknown.stage = "manual_confirmation_state_unknown";
  unknown.ambiguous = true;
  throw unknown;
}

module.exports = {
  IDS: IDS,
  preview: preview,
  submit: submit,
  fillOrderFields: fillOrderFields,
  ensureTradingPage: ensureTradingPage,
  recoverToTradingPage: recoverToTradingPage
};
