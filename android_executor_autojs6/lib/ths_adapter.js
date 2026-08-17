"use strict";

// Resource IDs are based on the open-source thsauto project's Android THS
// implementation. They MUST be verified against the installed THS version on
// the target phone before real-money use.
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

function waitId(resourceId, timeoutMs) {
  var node = id(resourceId).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS UI node not found: " + resourceId);
  return node;
}

function clickTradeTab(label, timeoutMs) {
  var node = id(IDS.navButton).text(label).findOne(Number(timeoutMs || 5000));
  if (!node) throw new Error("THS trading tab not found: " + label);
  if (!node.click()) {
    var b = node.bounds();
    click(b.centerX(), b.centerY());
  }
}

function clearAndSet(node, value) {
  if (!node.setText(String(value))) {
    node.click();
    sleep(150);
    setText(String(value));
  }
}

function ensureTradingPage(side, config) {
  app.launchPackage(String(config.ths_package || "com.hexin.plat.android"));
  sleep(400);
  var codeNode = id(IDS.stockCode).findOne(1200);
  if (!codeNode) {
    // thsauto also expects the user to have entered the trading interface first.
    // We only try the visible trading tab and deliberately fail instead of
    // guessing screen coordinates if the trading panel is absent.
    clickTradeTab(side === "buy" ? "买入" : "卖出", config.ui_timeout_ms);
    codeNode = id(IDS.stockCode).findOne(Number(config.ui_timeout_ms || 5000));
  }
  if (!codeNode) throw new Error("not on THS trading page; open 交易 -> A股/模拟 first");
  return codeNode;
}

function fillCode(code, config) {
  var codeNode = waitId(IDS.stockCode, config.ui_timeout_ms);
  codeNode.click();
  sleep(150);
  var edit = id(IDS.searchContainer).className("android.widget.EditText").findOne(Number(config.ui_timeout_ms || 5000));
  if (!edit) throw new Error("THS stock search EditText not found");
  clearAndSet(edit, code);
  sleep(500);
  var suggestion = id(IDS.stockSuggestionCode).text(code).findOne(Number(config.ui_timeout_ms || 5000));
  if (!suggestion) throw new Error("THS stock suggestion does not match code=" + code);
  suggestion.click();
  sleep(250);
}

function inputOrder(order, config) {
  ensureTradingPage(order.side, config);
  clickTradeTab(order.side === "buy" ? "买入" : "卖出", config.ui_timeout_ms);
  sleep(250);
  fillCode(order.code, config);

  var volumeEdit = id(IDS.stockVolume).className("android.widget.EditText").findOne(Number(config.ui_timeout_ms || 5000));
  if (!volumeEdit) throw new Error("THS volume EditText not found");
  clearAndSet(volumeEdit, String(order.qty));

  var priceEdit = id(IDS.stockPrice).className("android.widget.EditText").findOne(Number(config.ui_timeout_ms || 5000));
  if (!priceEdit) throw new Error("THS price EditText not found");
  clearAndSet(priceEdit, Number(order.submit_price).toFixed(2));

  var submit = waitId(IDS.transaction, config.ui_timeout_ms);
  submit.click();
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
  var cancelNode = id(IDS.cancel).findOne(Number(config.ui_timeout_ms || 5000));
  var okNode = id(IDS.ok).findOne(Number(config.ui_timeout_ms || 5000));
  if (cancelNode && okNode) {
    // Dry-run contract: reach the confirmation dialog, then cancel.
    cancelNode.click();
    return { success: true, mode: "dry_run", stage: "confirmation_reached" };
  }
  var prompt = readPrompt();
  var finalOk = id(IDS.ok).findOne(300);
  if (finalOk) finalOk.click();
  throw new Error("THS did not show order confirmation: " + JSON.stringify(prompt));
}

function submit(order, config) {
  inputOrder(order, config);
  var okNode = id(IDS.ok).findOne(Number(config.ui_timeout_ms || 5000));
  var cancelNode = id(IDS.cancel).findOne(500);
  if (!okNode || !cancelNode) {
    var badPrompt = readPrompt();
    if (okNode) okNode.click();
    throw new Error("THS order confirmation missing: " + JSON.stringify(badPrompt));
  }
  okNode.click();
  sleep(500);
  var prompt = readPrompt();
  var finalOk = id(IDS.ok).findOne(Number(config.ui_timeout_ms || 5000));
  if (finalOk) finalOk.click();
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
  ensureTradingPage: ensureTradingPage
};
