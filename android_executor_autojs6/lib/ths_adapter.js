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
var THS_PACKAGE = "com.hexin.plat.android";
var TAP_ATTEMPTS = 3;
var PROBE_MAX_NODES = 400;

function timeout(config) {
  return Number((config && config.ui_timeout_ms) || 5000);
}

function waitId(resourceId, timeoutMs) {
  var node = id(resourceId).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS UI node not found: " + resourceId);
  return node;
}

function clickNode(node) {
  if (!node) return false;
  if (node.click()) return true;
  var b = node.bounds();
  return click(b.centerX(), b.centerY());
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
  var b = target.bounds();
  return click(b.centerX(), b.centerY());
}

function safeNodeText(node) {
  try {
    return node ? String(node.text() || "") : "";
  } catch (e) {
    return "";
  }
}

function clearAndSet(node, value) {
  var v = String(value);
  try {
    node.setText(v);
  } catch (e) {
    // Fall through to focused setText below.
  }
  if (safeNodeText(node) === v) return;
  if (!clickNode(node)) throw new Error("THS input focus failed");
  sleep(120);
  if (!setText(v)) throw new Error("THS setText failed: " + v);
}

function findInside(container, selector, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var node = null;
    try { node = container.findOne(selector); } catch (e) { node = null; }
    if (node) return node;
    sleep(80);
  }
  return null;
}

function waitDescendantEdit(containerId, hint, timeoutMs) {
  var t = Number(timeoutMs || 5000);
  var container = waitId(containerId, t);
  var edit = findInside(container, className(EDIT_TEXT), t);
  if (!edit && hint) edit = className(EDIT_TEXT).text(String(hint)).findOne(400);
  if (!edit) throw new Error("THS descendant EditText not found: " + containerId);
  return edit;
}

function waitFieldValue(containerId, expected, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 1200);
  var expectedText = String(expected);
  while (Date.now() <= deadline) {
    try {
      var container = id(containerId).findOnce();
      if (container) {
        var edit = container.findOne(className(EDIT_TEXT));
        if (edit && safeNodeText(edit) === expectedText) return true;
      }
    } catch (e) {
      // Retry until deadline.
    }
    sleep(80);
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
    sleep(700);
  }
  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(1200);
  if (!codeEdit) {
    clickTradeTab(side === "sell" ? "卖出" : "买入", timeout(config));
    codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(timeout(config));
  }
  if (!codeEdit) throw new Error("not on THS trading page");
  return codeEdit;
}

function findSuggestionOnce(searchContainer, code) {
  var node = id(IDS.stockSuggestionCode).text(String(code)).findOnce();
  if (node) return node;
  try {
    return searchContainer.findOne(className("android.widget.TextView").text(String(code)));
  } catch (e) {
    return null;
  }
}

function isMainTradeCodeResolved(code) {
  // A resolved symbol is defined by the final trade form, not by the transient
  // search suggestion list. THS may auto-resolve a valid six-digit code and close
  // the search overlay without ever exposing stockcode_tv.
  if (id(IDS.searchContainer).findOnce()) return false;
  var codeNode = id(IDS.stockCode).className(EDIT_TEXT).text(String(code)).findOnce();
  var transaction = id(IDS.transaction).findOnce();
  return !!(codeNode && transaction);
}

function waitMainTradeCodeResolved(code, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 1200);
  while (Date.now() <= deadline) {
    if (isMainTradeCodeResolved(code)) return true;
    sleep(80);
  }
  return false;
}

function fillCode(code, config) {
  var expected = String(code);
  if (isMainTradeCodeResolved(expected)) return;

  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(timeout(config));
  if (!codeEdit) throw new Error("THS stock code EditText not found");
  if (!clickNode(codeEdit)) throw new Error("THS stock code click failed");
  sleep(180);

  var searchContainer = waitId(IDS.searchContainer, timeout(config));
  var searchEdit = findInside(searchContainer, id(IDS.stockCode).className(EDIT_TEXT), timeout(config));
  if (!searchEdit) throw new Error("THS stock search EditText not found");
  clearAndSet(searchEdit, expected);

  var deadline = Date.now() + timeout(config);
  while (Date.now() <= deadline) {
    // Preferred success condition: THS has already accepted the code and returned
    // to the main trade form. No suggestion click is required in this path.
    if (isMainTradeCodeResolved(expected)) return;

    var activeSearch = id(IDS.searchContainer).findOnce();
    if (activeSearch) {
      var suggestion = findSuggestionOnce(activeSearch, expected);
      if (suggestion) {
        for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
          if (!tapCenter(suggestion)) break;
          if (waitMainTradeCodeResolved(expected, 650)) return;
          activeSearch = id(IDS.searchContainer).findOnce();
          if (!activeSearch) break;
          suggestion = findSuggestionOnce(activeSearch, expected);
          if (!suggestion) break;
        }
      }
    }
    sleep(100);
  }

  // Final check avoids false failure when THS resolves the symbol exactly as the
  // timeout expires.
  if (isMainTradeCodeResolved(expected)) return;
  throw new Error("THS stock code not resolved on final trade form code=" + expected);
}

function fillOrderFields(order, config) {
  ensureTradingPage(order.side, config);
  clickTradeTab(order.side === "sell" ? "卖出" : "买入", timeout(config));
  sleep(220);
  fillCode(order.code, config);

  if (!isMainTradeCodeResolved(String(order.code))) {
    throw new Error("THS final code verification failed code=" + order.code);
  }

  var qtyText = String(order.qty);
  var priceText = Number(order.submit_price).toFixed(2);

  var volumeEdit = waitDescendantEdit(IDS.stockVolume, "数量", timeout(config));
  clearAndSet(volumeEdit, qtyText);
  if (!waitFieldValue(IDS.stockVolume, qtyText, 1200)) {
    volumeEdit = waitDescendantEdit(IDS.stockVolume, "数量", timeout(config));
    clearAndSet(volumeEdit, qtyText);
    if (!waitFieldValue(IDS.stockVolume, qtyText, 1200)) {
      throw new Error("THS quantity verification failed expected=" + qtyText);
    }
  }

  var priceEdit = waitDescendantEdit(IDS.stockPrice, "价格", timeout(config));
  clearAndSet(priceEdit, priceText);
  if (!waitFieldValue(IDS.stockPrice, priceText, 1200)) {
    priceEdit = waitDescendantEdit(IDS.stockPrice, "价格", timeout(config));
    clearAndSet(priceEdit, priceText);
    if (!waitFieldValue(IDS.stockPrice, priceText, 1200)) {
      throw new Error("THS price verification failed expected=" + priceText);
    }
  }

  return {
    success: true,
    stage: "fields_filled_and_verified",
    code: String(order.code),
    side: String(order.side),
    qty: Number(order.qty),
    submit_price: Number(order.submit_price)
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
    sleep(90);
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
    sleep(90);
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
  fillOrderFields(order, config);
  var waitMs = Math.max(900, Math.min(1800, Math.floor(timeout(config) / 2)));

  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    var existing = findConfirmation(order);
    if (existing) return existing;

    var submitNode = waitId(IDS.transaction, Math.min(timeout(config), 1500));
    if (!tapCenter(submitNode)) throw new Error("THS transaction button tap failed");

    var confirmation = waitConfirmation(order, waitMs);
    if (confirmation) return confirmation;
    sleep(160);
  }

  var probe = dumpProbe(order, "transaction_no_confirmation_after_retries");
  throw new Error("THS transaction tap did not open " + confirmationLabels(order.side).title + "; probe=" + probe);
}

function dismissSuccessDialog(order, dialog) {
  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    if (!textContains("委托已提交").findOnce()) return;
    var current = findSuccessDialog() || dialog;
    if (!current || !current.ok || !tapCenter(current.ok)) break;
    sleep(350);
    if (!textContains("委托已提交").findOnce()) return;
  }
  var probe = dumpProbe(order, "success_dialog_not_dismissed_after_retries");
  throw new Error("THS success dialog did not dismiss after retries; probe=" + probe);
}

function preview(order, config) {
  var confirmation = openOrderConfirmation(order, config);
  for (var attempt = 1; attempt <= TAP_ATTEMPTS; attempt++) {
    var current = findConfirmation(order);
    if (!current) return { success: true, mode: "dry_run", stage: "confirmation_reached_and_cancelled" };
    if (!tapCenter(current.cancel)) break;
    sleep(300);
  }
  var probe = dumpProbe(order, "cancel_confirmation_not_dismissed");
  throw new Error("THS confirmation cancel did not dismiss; probe=" + probe);
}

function submit(order, config) {
  openOrderConfirmation(order, config);
  var success = null;

  for (var attempt = 1; attempt <= 2; attempt++) {
    success = findSuccessDialog();
    if (success) break;

    var confirmation = findConfirmation(order);
    if (!confirmation) {
      success = waitSuccessDialog(timeout(config));
      break;
    }

    if (!tapCenter(confirmation.action)) throw new Error("THS order confirmation tap failed");
    success = waitSuccessDialog(Math.max(1800, Math.min(timeout(config), 3500)));
    if (success) break;

    // Retry final confirmation only if the exact confirmation dialog is still
    // present. If it disappeared, state is ambiguous; stop to avoid duplicates.
    if (!findConfirmation(order)) break;
    sleep(160);
  }

  if (!success) {
    var probe = dumpProbe(order, "confirmation_result_ambiguous");
    throw new Error("THS confirmation result is ambiguous or unrecognized; probe=" + probe);
  }

  dismissSuccessDialog(order, success);
  sleep(220);

  return {
    success: true,
    mode: "live",
    prompt: { title: success.title, content: success.message },
    contract_no: success.contract_no,
    stage: "submitted_and_result_dismissed"
  };
}

module.exports = {
  IDS: IDS,
  preview: preview,
  submit: submit,
  fillOrderFields: fillOrderFields,
  ensureTradingPage: ensureTradingPage
};
