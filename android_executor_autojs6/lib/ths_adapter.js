"use strict";

// Resource IDs verified against the target phone's THS trading page on 2026-08-17.
// Price and volume IDs identify FrameLayout containers; their editable controls
// are descendant android.widget.EditText nodes without their own resource IDs.
var IDS = {
  navButton: "com.hexin.plat.android:id/btn",
  stockCode: "com.hexin.plat.android:id/auto_stockcode",
  searchContainer: "com.hexin.plat.android:id/dialogplus_view_container",
  stockSuggestionCode: "com.hexin.plat.android:id/stockcode_tv",
  stockVolume: "com.hexin.plat.android:id/stockvolume",
  stockPrice: "com.hexin.plat.android:id/stockprice",
  transaction: "com.hexin.plat.android:id/btn_transaction",
  ok: "com.hexin.plat.android:id/ok_btn",
  cancel: "com.hexin.plat.android:id/cancel_btn",
  prompt: "com.hexin.plat.android:id/prompt_content",
  dialogTitle: "com.hexin.plat.android:id/dialog_title"
};

var EDIT_TEXT = "android.widget.EditText";

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

function waitEditById(resourceId, timeoutMs) {
  var node = id(resourceId).className(EDIT_TEXT).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS EditText not found: " + resourceId);
  return node;
}

function findInside(container, selector, timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var node = null;
    try {
      node = container.findOne(selector);
    } catch (e) {
      node = null;
    }
    if (node) return node;
    sleep(100);
  }
  return null;
}

function waitDescendantEdit(containerId, hint, timeoutMs) {
  var limit = Number(timeoutMs || 5000);
  var container = waitId(containerId, limit);
  var edit = findInside(container, className(EDIT_TEXT), limit);
  if (!edit && hint) {
    edit = className(EDIT_TEXT).text(String(hint)).findOne(500);
  }
  if (!edit) {
    throw new Error("THS descendant EditText not found: " + containerId + " hint=" + hint);
  }
  return edit;
}

function clickTradeTab(label, timeoutMs) {
  var node = id(IDS.navButton).text(label).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS trading tab not found: " + label);
  if (!clickNode(node)) throw new Error("THS trading tab click failed: " + label);
}

function clearAndSet(node, value) {
  var textValue = String(value);
  if (node.setText(textValue)) return;
  clickNode(node);
  sleep(150);
  if (!setText(textValue)) throw new Error("THS setText failed: " + textValue);
}

function ensureTradingPage(side, config) {
  var pkg = String(config.ths_package || "com.hexin.plat.android");
  if (currentPackage() !== pkg) {
    app.launchPackage(pkg);
    sleep(700);
  }

  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(1200);
  if (!codeEdit) {
    // If THS was brought to foreground on an already-open trading screen, select
    // the requested side. We deliberately do not guess coordinates or navigate
    // account/login screens.
    clickTradeTab(side === "buy" ? "买入" : "卖出", timeout(config));
    codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(timeout(config));
  }
  if (!codeEdit) throw new Error("not on THS trading page; open 交易 -> 买入/卖出 first");
  return codeEdit;
}

function findSuggestion(searchContainer, code, timeoutMs) {
  var limit = Number(timeoutMs || 5000);
  var deadline = Date.now() + limit;
  while (Date.now() <= deadline) {
    var suggestion = id(IDS.stockSuggestionCode).text(String(code)).findOnce();
    if (suggestion) return suggestion;

    // Some THS builds expose the result code only as TextView text without the
    // historical stockcode_tv resource id. Restrict the fallback to the active
    // search dialog so the typed EditText itself cannot be mistaken for a result.
    suggestion = findInside(
      searchContainer,
      className("android.widget.TextView").text(String(code)),
      150
    );
    if (suggestion) return suggestion;
    sleep(100);
  }
  return null;
}

function fillCode(code, config) {
  var codeEdit = waitEditById(IDS.stockCode, timeout(config));
  if (!clickNode(codeEdit)) throw new Error("THS stock code field click failed");
  sleep(200);

  var searchContainer = waitId(IDS.searchContainer, timeout(config));
  var searchEdit = findInside(
    searchContainer,
    id(IDS.stockCode).className(EDIT_TEXT),
    timeout(config)
  );
  if (!searchEdit) throw new Error("THS stock search EditText not found");

  clearAndSet(searchEdit, code);
  sleep(350);

  var suggestion = findSuggestion(searchContainer, code, timeout(config));
  if (!suggestion) throw new Error("THS stock suggestion does not match code=" + code);
  if (!clickNode(suggestion)) throw new Error("THS stock suggestion click failed code=" + code);
  sleep(350);
}

function fillOrderFields(order, config) {
  ensureTradingPage(order.side, config);
  clickTradeTab(order.side === "buy" ? "买入" : "卖出", timeout(config));
  sleep(250);
  fillCode(order.code, config);

  var volumeEdit = waitDescendantEdit(IDS.stockVolume, "数量", timeout(config));
  clearAndSet(volumeEdit, String(order.qty));

  var priceEdit = waitDescendantEdit(IDS.stockPrice, "价格", timeout(config));
  clearAndSet(priceEdit, Number(order.submit_price).toFixed(2));

  return {
    success: true,
    stage: "fields_filled",
    code: String(order.code),
    side: String(order.side),
    qty: Number(order.qty),
    submit_price: Number(order.submit_price)
  };
}

function inputOrder(order, config) {
  fillOrderFields(order, config);

  var submit = waitId(IDS.transaction, timeout(config));
  if (!clickNode(submit)) throw new Error("THS transaction button click failed");
  sleep(300);
}

function readPrompt() {
  var titleNode = id(IDS.dialogTitle).findOne(400);
  var promptNode = id(IDS.prompt).findOne(800);
  return {
    title: titleNode ? String(titleNode.text()) : "",
    content: promptNode ? String(promptNode.text()) : ""
  };
}

function preview(order, config) {
  inputOrder(order, config);
  var cancelNode = id(IDS.cancel).findOne(timeout(config));
  var okNode = id(IDS.ok).findOne(800);
  if (cancelNode && okNode) {
    // DRY-RUN safety contract: reaching the broker confirmation is success only
    // after clicking the explicit cancel control. Never click OK in dry-run.
    if (!clickNode(cancelNode)) throw new Error("THS confirmation cancel click failed");
    return { success: true, mode: "dry_run", stage: "confirmation_reached_and_cancelled" };
  }

  var prompt = readPrompt();
  // Fail closed. The confirmation dialog may remain visible for manual inspection,
  // but no confirm/OK control is touched when its structure is not recognized.
  throw new Error(
    "THS confirmation controls not recognized; no confirm button clicked: " +
    JSON.stringify(prompt)
  );
}

function submit(order, config) {
  inputOrder(order, config);
  var okNode = id(IDS.ok).findOne(timeout(config));
  var cancelNode = id(IDS.cancel).findOne(500);
  if (!okNode || !cancelNode) {
    var badPrompt = readPrompt();
    throw new Error("THS order confirmation missing: " + JSON.stringify(badPrompt));
  }

  if (!clickNode(okNode)) throw new Error("THS order confirmation click failed");
  sleep(500);
  var prompt = readPrompt();
  var finalOk = id(IDS.ok).findOne(timeout(config));
  if (finalOk) clickNode(finalOk);
  var content = String(prompt.content || "");
  if (content.indexOf("委托已提交") < 0 && content.indexOf("合同号") < 0) {
    throw new Error("THS order not confirmed submitted: " + JSON.stringify(prompt));
  }
  return { success: true, mode: "live", prompt: prompt };
}

module.exports = {
  IDS: IDS,
  preview: preview,
  submit: submit,
  fillOrderFields: fillOrderFields,
  ensureTradingPage: ensureTradingPage
};
