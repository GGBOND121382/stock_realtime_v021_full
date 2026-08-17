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
var THS_PACKAGE = "com.hexin.plat.android";
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

function clickNodeCenter(node) {
  if (!node) return false;
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
  var pkg = String(config.ths_package || THS_PACKAGE);
  if (currentPackage() !== pkg) {
    app.launchPackage(pkg);
    sleep(700);
  }

  var codeEdit = id(IDS.stockCode).className(EDIT_TEXT).findOne(1200);
  if (!codeEdit) {
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

function confirmationLabels(side) {
  return side === "sell" ? {
    title: "委托卖出确认",
    action: "确认卖出"
  } : {
    title: "委托买入确认",
    action: "确认买入"
  };
}

function waitConfirmation(order, timeoutMs) {
  var labels = confirmationLabels(order.side);
  var deadline = Date.now() + Number(timeoutMs || 5000);
  while (Date.now() <= deadline) {
    var titleNode = text(labels.title).findOnce();
    var actionNode = text(labels.action).findOnce();
    var cancelNode = text("取消").findOnce();
    if (titleNode && actionNode && cancelNode) {
      return {
        title: titleNode,
        action: actionNode,
        cancel: cancelNode,
        labels: labels
      };
    }
    sleep(100);
  }
  return null;
}

function safeValue(fn, fallback) {
  try {
    var value = fn();
    return value === null || value === undefined ? fallback : value;
  } catch (e) {
    return fallback;
  }
}

function clean(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t");
}

function timestampToken() {
  var d = new Date();
  function pad(n) { return n < 10 ? "0" + n : String(n); }
  return d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "_" +
    pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
}

function nodeLine(node, index) {
  var pkg = safeValue(function () { return node.packageName(); }, "");
  var rid = safeValue(function () { return node.id(); }, "");
  var textValue = safeValue(function () { return node.text(); }, "");
  var desc = safeValue(function () { return node.desc(); }, "");
  var cls = safeValue(function () { return node.className(); }, "");
  var clickable = safeValue(function () { return node.clickable(); }, false);
  var enabled = safeValue(function () { return node.enabled(); }, false);
  var bounds = safeValue(function () { return node.bounds(); }, null);
  return "[" + index + "]" +
    " package=" + clean(pkg) +
    " id=" + clean(rid) +
    " text=" + JSON.stringify(clean(textValue)) +
    " desc=" + JSON.stringify(clean(desc)) +
    " class=" + clean(cls) +
    " clickable=" + clickable +
    " enabled=" + enabled +
    " bounds=" + (bounds ? clean(bounds.toString()) : "");
}

function dumpPostSubmitProbe(order, stage) {
  var lines = [
    "========== THS POST-SUBMIT PROBE ==========",
    "stage=" + String(stage || "unknown"),
    "order=" + JSON.stringify(order || {}),
    "currentPackage=" + safeValue(function () { return currentPackage(); }, ""),
    "currentActivity=" + safeValue(function () { return currentActivity(); }, ""),
    "known_ok_count=" + id(IDS.ok).find().size(),
    "known_cancel_count=" + id(IDS.cancel).find().size(),
    "known_prompt_count=" + id(IDS.prompt).find().size(),
    "known_title_count=" + id(IDS.dialogTitle).find().size(),
    "---------- accessibility nodes ----------"
  ];

  var nodes = null;
  try {
    nodes = classNameMatches(/.*/).find();
  } catch (e) {
    lines.push("[WARN] cannot enumerate nodes: " + String(e));
  }

  if (nodes) {
    var limit = Math.min(nodes.size(), PROBE_MAX_NODES);
    for (var i = 0; i < limit; i++) {
      var node = nodes.get(i);
      var pkg = safeValue(function () { return node.packageName(); }, "");
      var rid = safeValue(function () { return node.id(); }, "");
      var textValue = safeValue(function () { return node.text(); }, "");
      var desc = safeValue(function () { return node.desc(); }, "");
      var clickable = safeValue(function () { return node.clickable(); }, false);
      if (String(pkg) === THS_PACKAGE || rid || textValue || desc || clickable) {
        lines.push(nodeLine(node, i));
      }
    }
    if (nodes.size() > PROBE_MAX_NODES) {
      lines.push("[TRUNCATED] total_nodes=" + nodes.size() + " max_nodes=" + PROBE_MAX_NODES);
    }
  }

  lines.push("========== END THS POST-SUBMIT PROBE ==========");
  var path = files.join(files.cwd(), "ths_order_confirmation_probe_" + timestampToken() + ".txt");
  files.write(path, lines.join("\n"));
  return path;
}

function openOrderConfirmation(order, config) {
  fillOrderFields(order, config);

  var submit = waitId(IDS.transaction, timeout(config));
  // THS's transaction container may report accessibility ACTION_CLICK success
  // without changing the UI. Use a physical screen-coordinate tap on the verified
  // transaction button bounds, then require the current confirmation dialog text.
  if (!clickNodeCenter(submit)) {
    throw new Error("THS transaction button coordinate tap failed");
  }

  var confirmation = waitConfirmation(order, timeout(config));
  if (!confirmation) {
    var probePath = dumpPostSubmitProbe(order, "transaction_tap_no_confirmation");
    throw new Error(
      "THS transaction tap did not open " + confirmationLabels(order.side).title +
      "; probe=" + probePath
    );
  }
  return confirmation;
}

function extractContractNumber(message) {
  var textValue = String(message || "");
  var match = textValue.match(/合同号(?:为)?[：:\s]*([0-9\s]+)/);
  return match ? String(match[1]).replace(/\s/g, "") : "";
}

function waitSuccessDialog(timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 3000);
  while (Date.now() <= deadline) {
    var messageNode = textContains("委托已提交").findOnce();
    var okNode = text("确定").findOnce();
    var titleNode = text("系统信息").findOnce();
    if (messageNode && okNode) {
      var message = String(messageNode.text() || "");
      return {
        title: titleNode ? String(titleNode.text() || "") : "系统信息",
        message: message,
        contract_no: extractContractNumber(message),
        ok: okNode
      };
    }
    sleep(100);
  }
  return null;
}

function waitSuccessDialogDismissed(timeoutMs) {
  var deadline = Date.now() + Number(timeoutMs || 2000);
  while (Date.now() <= deadline) {
    if (!textContains("委托已提交").findOnce()) return true;
    sleep(100);
  }
  return false;
}

function preview(order, config) {
  var confirmation = openOrderConfirmation(order, config);
  if (!clickNodeCenter(confirmation.cancel)) {
    throw new Error("THS confirmation cancel coordinate tap failed");
  }
  return {
    success: true,
    mode: "dry_run",
    stage: "confirmation_reached_and_cancelled"
  };
}

function submit(order, config) {
  var confirmation = openOrderConfirmation(order, config);
  if (!clickNodeCenter(confirmation.action)) {
    throw new Error("THS order confirmation coordinate tap failed");
  }

  var successDialog = waitSuccessDialog(timeout(config));
  if (!successDialog) {
    var resultProbe = dumpPostSubmitProbe(order, "confirmation_clicked_result_unrecognized");
    throw new Error(
      "THS confirmation was tapped but submission result is not recognized; probe=" + resultProbe
    );
  }

  if (!clickNodeCenter(successDialog.ok)) {
    var okProbe = dumpPostSubmitProbe(order, "success_dialog_ok_tap_failed");
    throw new Error("THS success dialog OK coordinate tap failed; probe=" + okProbe);
  }
  if (!waitSuccessDialogDismissed(2000)) {
    var dismissProbe = dumpPostSubmitProbe(order, "success_dialog_not_dismissed");
    throw new Error("THS success dialog did not dismiss after OK tap; probe=" + dismissProbe);
  }

  return {
    success: true,
    mode: "live",
    prompt: {
      title: successDialog.title,
      content: successDialog.message
    },
    contract_no: successDialog.contract_no,
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
